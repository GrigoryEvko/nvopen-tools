// Function: sub_DCF230
// Address: 0xdcf230
//
__int64 __fastcall sub_DCF230(__int64 *a1, char *a2, __int64 *a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // r15
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rax
  char *v15; // rdx
  _BYTE *v17; // [rsp+20h] [rbp-60h] BYREF
  __int64 v18; // [rsp+28h] [rbp-58h]
  _BYTE v19[80]; // [rsp+30h] [rbp-50h] BYREF

  result = a1[18];
  if ( !result )
  {
    v5 = *a1;
    v17 = v19;
    v18 = 0x400000000LL;
    v7 = v5 + 112LL * *((unsigned int *)a1 + 2);
    if ( v7 == v5 )
      goto LABEL_15;
    do
    {
      while ( 1 )
      {
        v8 = *(_QWORD *)(v5 + 56);
        if ( !sub_D96A50(v8) )
        {
          v12 = (unsigned int)v18;
          v9 = HIDWORD(v18);
          v13 = (unsigned int)v18 + 1LL;
          if ( v13 > HIDWORD(v18) )
          {
            a2 = v19;
            sub_C8D5F0((__int64)&v17, v19, v13, 8u, v10, v11);
            v12 = (unsigned int)v18;
          }
          *(_QWORD *)&v17[8 * v12] = v8;
          LODWORD(v18) = v18 + 1;
          if ( a4 )
            break;
        }
        v5 += 112;
        if ( v5 == v7 )
          goto LABEL_10;
      }
      v14 = *(unsigned int *)(v5 + 72);
      v15 = *(char **)(v5 + 64);
      v5 += 112;
      a2 = (char *)(*(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8));
      sub_D91A50(a4, a2, v15, &v15[8 * v14]);
    }
    while ( v5 != v7 );
LABEL_10:
    if ( !(_DWORD)v18 )
    {
LABEL_15:
      a1[18] = sub_D970F0((__int64)a3);
    }
    else
    {
      a2 = (char *)&v17;
      a1[18] = (__int64)sub_DCEEE0(a3, (__int64)&v17, 1, v9, v10);
    }
    if ( v17 != v19 )
      _libc_free(v17, a2);
    return a1[18];
  }
  return result;
}
