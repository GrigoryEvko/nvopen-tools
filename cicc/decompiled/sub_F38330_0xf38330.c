// Function: sub_F38330
// Address: 0xf38330
//
unsigned __int64 __fastcall sub_F38330(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        unsigned __int64 *a4,
        unsigned __int64 *a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  unsigned __int64 v10; // rax
  int v11; // edx
  unsigned __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  int v15; // edx
  unsigned __int64 result; // rax
  __int64 v17; // [rsp+0h] [rbp-20h] BYREF
  __int64 v18[3]; // [rsp+8h] [rbp-18h] BYREF

  v17 = 0;
  v18[0] = 0;
  sub_F377A0(a1, a2, a3, &v17, v18, 0, 0, a6, a7, a8);
  v10 = *(_QWORD *)(v17 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v10 == v17 + 48 )
  {
    v12 = 0;
  }
  else
  {
    if ( !v10 )
      BUG();
    v11 = *(unsigned __int8 *)(v10 - 24);
    v12 = v10 - 24;
    if ( (unsigned int)(v11 - 30) >= 0xB )
      v12 = 0;
  }
  v13 = v18[0];
  *a4 = v12;
  v14 = *(_QWORD *)(v13 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v14 == v13 + 48 )
  {
    result = 0;
  }
  else
  {
    if ( !v14 )
      BUG();
    v15 = *(unsigned __int8 *)(v14 - 24);
    result = v14 - 24;
    if ( (unsigned int)(v15 - 30) >= 0xB )
      result = 0;
  }
  *a5 = result;
  return result;
}
