// Function: sub_2F60D70
// Address: 0x2f60d70
//
__int64 __fastcall sub_2F60D70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // r13
  char v9; // r15
  unsigned __int8 v10; // al
  bool v11; // zf
  char v12; // dl
  unsigned int v13; // ecx

  v6 = *(_QWORD *)a3;
  v7 = *(_QWORD *)a3 + 24LL * *(unsigned int *)(a3 + 8);
  if ( v7 == *(_QWORD *)a3 )
  {
    v12 = 0;
    v10 = 0;
  }
  else
  {
    v9 = 0;
    v10 = 0;
    do
    {
      while ( *(_QWORD *)(v6 + 16) != a4 )
      {
        v6 += 24;
        if ( v7 == v6 )
          goto LABEL_8;
      }
      v11 = (((unsigned __int8)*(_QWORD *)(sub_2E0F080(a1, a2, a3, *(_QWORD *)(v6 + 8), a5, a6, *(_OWORD *)v6, a2) + 8)
            ^ 6)
           & 6) == 0;
      v10 = 1;
      if ( v11 )
        v9 = 1;
      v6 += 24;
    }
    while ( v7 != v6 );
LABEL_8:
    v12 = v9;
  }
  v13 = v10;
  BYTE1(v13) = v12;
  return v13;
}
