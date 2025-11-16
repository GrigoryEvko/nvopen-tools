// Function: sub_6214E0
// Address: 0x6214e0
//
__int64 __fastcall sub_6214E0(__int16 *a1, int a2, int a3, int a4)
{
  int v5; // eax
  int v6; // r8d
  int v7; // r10d
  unsigned __int16 v8; // r9
  __int16 *v9; // rsi
  __int16 *v10; // r12
  unsigned int v11; // edx
  __int64 v12; // rbx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rdi
  __int64 v15; // rax
  __int64 result; // rax

  v5 = a2 / 16;
  v6 = a2 % 16;
  v7 = 16 - a2 % 16;
  if ( a3 && a4 )
    v8 = *a1 >> 15;
  else
    v8 = 0;
  v9 = a1;
  v10 = a1 - 8;
  v11 = 6 - v5;
  v12 = 7 - v5;
  do
  {
    v13 = v8;
    if ( v11 + 1 <= 7 )
      v13 = (unsigned __int16)v9[v12];
    v14 = v13 >> v6;
    v15 = v8;
    if ( v11 <= 7 )
      v15 = (unsigned __int16)a1[v11];
    --v9;
    --v11;
    result = v14 | (v15 << v7);
    v9[8] = result;
  }
  while ( v9 != v10 );
  return result;
}
