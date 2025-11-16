// Function: sub_2AB8CE0
// Address: 0x2ab8ce0
//
void __fastcall sub_2AB8CE0(
        __int8 *a1,
        size_t a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        unsigned __int8 **a7)
{
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r9
  __int64 v15; // [rsp+8h] [rbp-2A8h]
  __int64 v16; // [rsp+8h] [rbp-2A8h]
  __int64 v17; // [rsp+8h] [rbp-2A8h]
  __int64 v19; // [rsp+28h] [rbp-288h]
  unsigned __int8 *v20; // [rsp+38h] [rbp-278h] BYREF
  __m128i v21; // [rsp+40h] [rbp-270h] BYREF
  _BYTE v22[128]; // [rsp+50h] [rbp-260h] BYREF
  unsigned __int8 *v23[10]; // [rsp+D0h] [rbp-1E0h] BYREF
  char v24[400]; // [rsp+120h] [rbp-190h] BYREF

  sub_31A4FD0(v22, a6, 1, a5, 0);
  v20 = *a7;
  if ( v20 )
    sub_2AAAFA0((__int64 *)&v20);
  v19 = sub_31A4B60(v22);
  v13 = **(_QWORD **)(a6 + 32);
  if ( !v20 )
  {
    v16 = **(_QWORD **)(a6 + 32);
    sub_D4BD20(v23, a6, v10, v11, v12, v13);
    sub_9C6650(&v20);
    v14 = v16;
    v20 = v23[0];
    if ( v23[0] )
    {
      sub_B976B0((__int64)v23, v23[0], (__int64)&v20);
      v14 = v16;
      v23[0] = 0;
    }
    v17 = v14;
    sub_9C6650(v23);
    v13 = v17;
  }
  v15 = v13;
  sub_B157E0((__int64)&v21, &v20);
  sub_B17850((__int64)v23, v19, a3, a4, &v21, v15);
  sub_B18290((__int64)v23, a1, a2);
  sub_1049740(a5, (__int64)v23);
  v23[0] = (unsigned __int8 *)&unk_49D9D40;
  sub_23FD590((__int64)v24);
  if ( v20 )
    sub_B91220((__int64)&v20, (__int64)v20);
}
