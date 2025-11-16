// Function: sub_F38250
// Address: 0xf38250
//
unsigned __int64 __fastcall sub_F38250(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        char a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  unsigned __int64 v8; // rax
  int v9; // edx
  unsigned __int64 result; // rax

  sub_F377A0(a1, a2, a3, &a8, 0, a4, 0, a5, a6, a7);
  v8 = *(_QWORD *)(a8 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v8 == a8 + 48 )
    return 0;
  if ( !v8 )
    BUG();
  v9 = *(unsigned __int8 *)(v8 - 24);
  result = v8 - 24;
  if ( (unsigned int)(v9 - 30) >= 0xB )
    return 0;
  return result;
}
