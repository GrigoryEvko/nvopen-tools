// Function: sub_16E9180
// Address: 0x16e9180
//
__int64 __fastcall sub_16E9180(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  signed __int64 v9; // r14
  signed __int64 v10; // rcx
  signed __int64 v11; // rdx
  signed __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rsi
  _QWORD *v15; // rax
  __int64 v16; // r13
  __int64 v17; // rdx
  __int64 result; // rax

  v9 = a1[5];
  v10 = a1[4];
  v11 = v9;
  if ( v9 >= v10 )
  {
    v12 = ((v10 + 1 + ((unsigned __int64)(v10 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL) + (v10 + 1) / 2;
    if ( v10 < v12 )
    {
      sub_16E90A0((__int64)a1, v12, v9, v10, a5, a6);
      v11 = a1[5];
    }
  }
  v13 = a1[3];
  a1[5] = v11 + 1;
  *(_QWORD *)(v13 + 8 * v11) = a3 | a2;
  v14 = a1[3];
  v15 = a1 + 9;
  v16 = *(_QWORD *)(v14 + 8 * v9);
  do
  {
    if ( *v15 >= a4 )
      ++*v15;
    v17 = v15[10];
    if ( a4 <= v17 )
      v15[10] = v17 + 1;
    ++v15;
  }
  while ( v15 != a1 + 18 );
  memmove((void *)(v14 + 8 * a4 + 8), (const void *)(v14 + 8 * a4), 8 * (a1[5] + ~a4));
  result = a1[3];
  *(_QWORD *)(result + 8 * a4) = v16;
  return result;
}
