// Function: sub_25F76A0
// Address: 0x25f76a0
//
__int64 __fastcall sub_25F76A0(char *a1, char *a2, char *a3, char *a4, __int64 a5)
{
  char *v9; // rbx
  __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // r13
  unsigned __int64 v15; // rbx
  __int64 v16; // r14
  __int64 v17; // rsi
  __int64 v18; // rdi
  __int64 v20; // r14
  unsigned __int64 v21; // r13
  __int64 v22; // r15
  __int64 v23; // rsi
  __int64 v24; // rdi

  if ( a1 == a2 )
  {
LABEL_7:
    v14 = a4 - a3;
    v15 = 0x86BCA1AF286BCA1BLL * (v14 >> 3);
    if ( v14 > 0 )
    {
      v16 = a5;
      do
      {
        v17 = (__int64)a3;
        v18 = v16;
        a3 += 152;
        v16 += 152;
        sub_25F6310(v18, v17);
        --v15;
      }
      while ( v15 );
      return a5 + v14;
    }
    return a5;
  }
  v9 = a1;
  while ( a4 != a3 )
  {
    if ( *(_DWORD *)a3 < *(_DWORD *)v9 )
    {
      v10 = (__int64)a3;
      v11 = a5;
      a3 += 152;
      a5 += 152;
      sub_25F6310(v11, v10);
      if ( v9 == a2 )
        goto LABEL_7;
    }
    else
    {
      v12 = (__int64)v9;
      v13 = a5;
      v9 += 152;
      a5 += 152;
      sub_25F6310(v13, v12);
      if ( v9 == a2 )
        goto LABEL_7;
    }
  }
  v20 = a2 - v9;
  v21 = 0x86BCA1AF286BCA1BLL * (v20 >> 3);
  if ( v20 <= 0 )
    return a5;
  v22 = a5;
  do
  {
    v23 = (__int64)v9;
    v24 = v22;
    v9 += 152;
    v22 += 152;
    sub_25F6310(v24, v23);
    --v21;
  }
  while ( v21 );
  return a5 + v20;
}
