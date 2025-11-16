// Function: sub_22BA640
// Address: 0x22ba640
//
__int64 __fastcall sub_22BA640(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rsi
  unsigned __int64 v8; // r12
  __int64 v9; // r13
  __int64 v10; // r15
  char **v11; // rbx
  int v12; // eax
  char **v13; // rsi
  __int64 v14; // rdi

  v6 = a2 - a1;
  v8 = 0x8E38E38E38E38E39LL * (v6 >> 3);
  if ( v6 <= 0 )
    return a3;
  v9 = v6;
  v10 = a3 + 8;
  v11 = (char **)(a1 + 8);
  do
  {
    v12 = *((_DWORD *)v11 - 2);
    v13 = v11;
    v14 = v10;
    v11 += 9;
    v10 += 72;
    *(_DWORD *)(v10 - 80) = v12;
    sub_22AD4A0(v14, v13, a3, a4, a5, a6);
    --v8;
  }
  while ( v8 );
  return a3 + v9;
}
