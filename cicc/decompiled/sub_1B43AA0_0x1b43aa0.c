// Function: sub_1B43AA0
// Address: 0x1b43aa0
//
__int64 __fastcall sub_1B43AA0(void **a1)
{
  unsigned __int64 v2; // rax
  __int64 v3; // rbx
  int v4; // eax
  unsigned int v5; // edx
  __int64 v6; // r13
  __int64 v7; // rsi
  char *v8; // rax
  const void *v9; // r14
  __int64 v10; // rdi
  unsigned __int64 v12; // [rsp+10h] [rbp-60h]
  unsigned int v13; // [rsp+1Ch] [rbp-54h]
  const void *v14; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-48h]
  const void *v16; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v17; // [rsp+38h] [rbp-38h]

  v2 = *((unsigned int *)a1 + 2);
  v12 = v2;
  if ( v2 > 1 )
  {
    qsort(*a1, (__int64)(8 * v2) >> 3, 8u, (__compar_fn_t)sub_1B424C0);
    v12 = *((unsigned int *)a1 + 2);
  }
  v3 = 1;
  if ( v12 == 1 )
  {
LABEL_18:
    LODWORD(v6) = 1;
  }
  else
  {
    while ( 1 )
    {
      v6 = 8 * v3;
      v7 = *((_QWORD *)*a1 + v3);
      v15 = *(_DWORD *)(v7 + 32);
      if ( v15 > 0x40 )
        sub_16A4FD0((__int64)&v14, (const void **)(v7 + 24));
      else
        v14 = *(const void **)(v7 + 24);
      sub_16A7490((__int64)&v14, 1);
      v8 = (char *)*a1;
      v5 = v15;
      v15 = 0;
      v9 = v14;
      v10 = *(_QWORD *)&v8[v6 - 8];
      v17 = v5;
      v16 = v14;
      if ( *(_DWORD *)(v10 + 32) > 0x40u )
      {
        v13 = v5;
        LOBYTE(v4) = sub_16A5220(v10 + 24, &v16);
        v5 = v13;
        LODWORD(v6) = v4;
      }
      else
      {
        LOBYTE(v6) = *(_QWORD *)(v10 + 24) == (_QWORD)v14;
      }
      if ( v5 > 0x40 )
      {
        if ( v9 )
        {
          j_j___libc_free_0_0(v9);
          if ( v15 > 0x40 )
          {
            if ( v14 )
              j_j___libc_free_0_0(v14);
          }
        }
      }
      if ( !(_BYTE)v6 )
        break;
      if ( v12 == ++v3 )
        goto LABEL_18;
    }
  }
  return (unsigned int)v6;
}
