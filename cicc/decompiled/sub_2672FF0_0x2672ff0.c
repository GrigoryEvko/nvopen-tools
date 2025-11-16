// Function: sub_2672FF0
// Address: 0x2672ff0
//
_QWORD *__fastcall sub_2672FF0(
        _QWORD *a1,
        __int64 **a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        int a8,
        int a9,
        __int64 a10,
        __int64 a11)
{
  __int64 v13; // rsi
  __int64 v14; // r14
  unsigned __int64 v15; // rdi
  int v16; // eax
  unsigned __int8 *v17; // rdi
  __int64 v18; // rax
  unsigned __int64 v19; // rdi
  int v20; // eax
  unsigned __int8 *v21; // rdi
  void *v23; // [rsp+0h] [rbp-50h] BYREF
  __int16 v24; // [rsp+20h] [rbp-30h]

  v13 = a11;
  v24 = 257;
  if ( a11 )
    v13 = a11 - 24;
  v14 = sub_F36960(a10, (__int64 *)(v13 + 24), 0, **a2, *a2[1], 0, &v23, 0);
  v15 = *(_QWORD *)(a10 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v15 == a10 + 48 )
  {
    v17 = 0;
  }
  else
  {
    if ( !v15 )
      BUG();
    v16 = *(unsigned __int8 *)(v15 - 24);
    v17 = (unsigned __int8 *)(v15 - 24);
    if ( (unsigned int)(v16 - 30) >= 0xB )
      v17 = 0;
  }
  sub_B46F90(v17, 0, *a2[2]);
  v18 = *a2[3];
  v19 = *(_QWORD *)(v18 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v19 == v18 + 48 )
  {
    v21 = 0;
  }
  else
  {
    if ( !v19 )
      BUG();
    v20 = *(unsigned __int8 *)(v19 - 24);
    v21 = (unsigned __int8 *)(v19 - 24);
    if ( (unsigned int)(v20 - 30) >= 0xB )
      v21 = 0;
  }
  sub_B46F90(v21, 0, v14);
  *a1 = 1;
  return a1;
}
