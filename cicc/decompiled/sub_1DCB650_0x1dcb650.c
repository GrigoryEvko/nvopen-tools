// Function: sub_1DCB650
// Address: 0x1dcb650
//
bool __fastcall sub_1DCB650(_QWORD *a1, __int64 a2, unsigned int a3, __int64 a4)
{
  _QWORD *v7; // rsi
  _QWORD *v8; // r8
  _QWORD *v10; // rax
  unsigned int v11; // ecx
  unsigned int v12; // edi
  unsigned int v13; // edx
  __int64 v14; // rax

  v7 = a1 + 1;
  v8 = (_QWORD *)a1[1];
  if ( v8 == a1 + 1 )
    goto LABEL_12;
  v10 = (_QWORD *)*a1;
  v11 = *(_DWORD *)(a2 + 48);
  if ( v7 == (_QWORD *)*a1 )
  {
    v10 = (_QWORD *)v10[1];
    v13 = v11 >> 7;
    *a1 = v10;
    v12 = *((_DWORD *)v10 + 4);
    if ( v11 >> 7 != v12 )
    {
LABEL_4:
      if ( v13 < v12 )
      {
        if ( v8 == v10 )
        {
          *a1 = v10;
LABEL_11:
          if ( *((_DWORD *)v10 + 4) != v13 )
            goto LABEL_12;
          goto LABEL_17;
        }
        do
          v10 = (_QWORD *)v10[1];
        while ( v8 != v10 && *((_DWORD *)v10 + 4) > v13 );
      }
      else
      {
        if ( v7 == v10 )
        {
LABEL_24:
          *a1 = v10;
          goto LABEL_12;
        }
        while ( v12 < v13 )
        {
          v10 = (_QWORD *)*v10;
          if ( v7 == v10 )
            goto LABEL_24;
          v12 = *((_DWORD *)v10 + 4);
        }
      }
      *a1 = v10;
      if ( v7 == v10 )
        goto LABEL_12;
      goto LABEL_11;
    }
    if ( v7 == v10 )
      goto LABEL_12;
  }
  else
  {
    v12 = *((_DWORD *)v10 + 4);
    v13 = v11 >> 7;
    if ( v11 >> 7 != v12 )
      goto LABEL_4;
  }
LABEL_17:
  if ( (v10[((v11 >> 6) & 1) + 3] & (1LL << v11)) != 0 )
    return 1;
LABEL_12:
  v14 = sub_1E69D00(a4, a3);
  return (!v14 || a2 != *(_QWORD *)(v14 + 24)) && sub_1DCB3F0((__int64)a1, a2) != 0;
}
