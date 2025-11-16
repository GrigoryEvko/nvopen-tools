// Function: sub_13436B0
// Address: 0x13436b0
//
__int64 *__fastcall sub_13436B0(_BYTE *a1, __int64 a2, unsigned int *a3, __int64 a4, __int64 *a5, _BYTE *a6)
{
  unsigned __int64 *v9; // rax
  unsigned __int64 *v10; // r15
  unsigned __int64 *v11; // r15
  __int64 v12; // rdi
  __int64 v13; // rdi
  char v16; // [rsp+Fh] [rbp-41h]
  __int64 v18; // [rsp+18h] [rbp-38h]

  v18 = a4 + 112;
  do
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v9 = (unsigned __int64 *)sub_1341B50((__int64)a1, *(_QWORD *)(a2 + 58384), a5, 0, *(_DWORD *)(a4 + 19424), 1);
        v10 = v9;
        if ( !v9 )
          break;
        sub_1342A40(v18, v9);
        if ( (unsigned __int8)sub_1343340(a1, a2, a3, (unsigned __int64 *)a5, v10) )
        {
          sub_1341570((__int64)a1, *(_QWORD *)(a2 + 58384), v10, *(_DWORD *)(a4 + 19424));
          v12 = a4 + 9768;
          if ( (*v10 & 0x10000) == 0 )
            v12 = v18;
          sub_1342830(v12, v10);
          break;
        }
        if ( *(_BYTE *)(a4 + 19432) )
        {
          *a6 = 1;
          return a5;
        }
        v11 = (unsigned __int64 *)sub_1341B50((__int64)a1, *(_QWORD *)(a2 + 58384), a5, 0, *(_DWORD *)(a4 + 19424), 0);
        if ( v11 )
        {
          v16 = 1;
          goto LABEL_12;
        }
      }
      v11 = (unsigned __int64 *)sub_1341B50((__int64)a1, *(_QWORD *)(a2 + 58384), a5, 0, *(_DWORD *)(a4 + 19424), 0);
      if ( !v11 )
        goto LABEL_18;
      v16 = 0;
LABEL_12:
      sub_1342A40(v18, v11);
      if ( (unsigned __int8)sub_1343340(a1, a2, a3, v11, a5) )
        break;
      if ( *(_BYTE *)(a4 + 19432) )
      {
        a5 = (__int64 *)v11;
        *a6 = 1;
        return a5;
      }
      a5 = (__int64 *)v11;
    }
    sub_1341570((__int64)a1, *(_QWORD *)(a2 + 58384), v11, *(_DWORD *)(a4 + 19424));
    v13 = a4 + 9768;
    if ( (*v11 & 0x10000) == 0 )
      v13 = v18;
    sub_1342830(v13, v11);
  }
  while ( v16 );
LABEL_18:
  if ( *(_BYTE *)(a4 + 19432) )
    *a6 = 0;
  return a5;
}
