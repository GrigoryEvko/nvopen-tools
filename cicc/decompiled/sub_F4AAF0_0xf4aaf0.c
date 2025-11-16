// Function: sub_F4AAF0
// Address: 0xf4aaf0
//
__int64 __fastcall sub_F4AAF0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v8; // rdx
  __int64 v9; // rdx

  v8 = *(_QWORD *)(a2 + 80);
  if ( !v8 )
    BUG();
  v9 = *(_QWORD *)(v8 + 32);
  if ( v9 )
    v9 -= 24;
  return sub_F49030(a1, a2, v9, a3, a4, a5, a6, a7);
}
