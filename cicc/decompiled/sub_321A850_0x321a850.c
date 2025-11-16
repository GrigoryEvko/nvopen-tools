// Function: sub_321A850
// Address: 0x321a850
//
void __fastcall sub_321A850(char **a1, char **a2)
{
  char **v2; // r12
  char *v4; // rax
  char **v5; // rbx
  char *v6; // r14
  unsigned __int64 *v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  char **v12; // r14
  __int64 v13; // rdx
  unsigned __int64 v14; // rbx
  __int64 v15; // rcx
  __int64 v16; // r15
  char *v17; // rdi
  char v19[8]; // [rsp+30h] [rbp-A0h] BYREF
  char *v20; // [rsp+38h] [rbp-98h]
  char *v21; // [rsp+50h] [rbp-80h] BYREF
  char *v22[2]; // [rsp+58h] [rbp-78h] BYREF
  _BYTE v23[48]; // [rsp+68h] [rbp-68h] BYREF
  char v24; // [rsp+98h] [rbp-38h]

  if ( a1 != a2 )
  {
    v2 = a1 + 10;
    while ( a2 != v2 )
    {
      v4 = *v2;
      v5 = v2;
      v2 += 10;
      sub_AF47B0((__int64)v19, *((unsigned __int64 **)v4 + 2), *((unsigned __int64 **)v4 + 3));
      v6 = v20;
      v7 = (unsigned __int64 *)*((_QWORD *)*a1 + 2);
      sub_AF47B0((__int64)&v21, v7, *((unsigned __int64 **)*a1 + 3));
      if ( v6 >= v22[0] )
      {
        sub_3219F60(v5, (__int64)v7, v8, v9, v10, v11);
      }
      else
      {
        v21 = *(v2 - 10);
        v22[0] = v23;
        v22[1] = (char *)0x200000000LL;
        if ( *((_DWORD *)v2 - 16) )
          sub_3218940((__int64)v22, v2 - 9, v8, v9, v10, v11);
        v12 = v2 - 9;
        v13 = (char *)v5 - (char *)a1;
        v24 = *((_BYTE *)v2 - 8);
        v14 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v5 - (char *)a1) >> 4);
        if ( v13 > 0 )
        {
          do
          {
            v15 = (__int64)*(v12 - 11);
            v16 = (__int64)v12;
            v12 -= 10;
            v12[9] = (char *)v15;
            sub_3218940(v16, v12, v13, v15, v10, v11);
            v9 = *(unsigned __int8 *)(v16 - 16);
            *(_BYTE *)(v16 + 64) = v9;
            --v14;
          }
          while ( v14 );
        }
        *a1 = v21;
        sub_3218940((__int64)(a1 + 1), v22, v13, v9, v10, v11);
        v17 = v22[0];
        *((_BYTE *)a1 + 72) = v24;
        if ( v17 != v23 )
          _libc_free((unsigned __int64)v17);
      }
    }
  }
}
