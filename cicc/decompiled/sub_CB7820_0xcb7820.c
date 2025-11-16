// Function: sub_CB7820
// Address: 0xcb7820
//
__int64 __fastcall sub_CB7820(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  signed __int64 v7; // r14
  signed __int64 v8; // rcx
  signed __int64 v9; // rdx
  signed __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rsi
  _QWORD *v13; // rax
  __int64 v14; // r13
  __int64 v15; // rdx
  __int64 result; // rax

  v7 = a1[5];
  v8 = a1[4];
  v9 = v7;
  if ( v7 >= v8 )
  {
    v10 = ((v8 + 1 + ((unsigned __int64)(v8 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL) + (v8 + 1) / 2;
    if ( v8 < v10 )
    {
      sub_CB7740((__int64)a1, v10);
      v9 = a1[5];
    }
  }
  v11 = a1[3];
  a1[5] = v9 + 1;
  *(_QWORD *)(v11 + 8 * v9) = a3 | a2;
  v12 = a1[3];
  v13 = a1 + 9;
  v14 = *(_QWORD *)(v12 + 8 * v7);
  do
  {
    if ( *v13 >= a4 )
      ++*v13;
    v15 = v13[10];
    if ( a4 <= v15 )
      v13[10] = v15 + 1;
    ++v13;
  }
  while ( v13 != a1 + 18 );
  memmove((void *)(v12 + 8 * a4 + 8), (const void *)(v12 + 8 * a4), 8 * (a1[5] + ~a4));
  result = a1[3];
  *(_QWORD *)(result + 8 * a4) = v14;
  return result;
}
