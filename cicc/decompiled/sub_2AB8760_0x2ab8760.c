// Function: sub_2AB8760
// Address: 0x2ab8760
//
void __fastcall sub_2AB8760(
        __int64 a1,
        __int64 a2,
        __int8 *a3,
        size_t a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rbx
  __int64 v18; // [rsp+18h] [rbp-288h]
  unsigned __int8 *v19; // [rsp+28h] [rbp-278h] BYREF
  __m128i v20; // [rsp+30h] [rbp-270h] BYREF
  _BYTE v21[128]; // [rsp+40h] [rbp-260h] BYREF
  unsigned __int8 *v22[10]; // [rsp+C0h] [rbp-1E0h] BYREF
  _BYTE v23[400]; // [rsp+110h] [rbp-190h] BYREF

  sub_31A4FD0(v21, a8, 1, a7, 0);
  v19 = 0;
  v18 = sub_31A4B60(v21);
  if ( a9 )
  {
    v15 = *(_QWORD *)(a9 + 40);
    if ( *(_QWORD *)(a9 + 48) )
    {
      sub_9C6650(&v19);
      v19 = *(unsigned __int8 **)(a9 + 48);
      if ( v19 )
        sub_2AAAFA0((__int64 *)&v19);
      goto LABEL_7;
    }
  }
  else
  {
    v15 = **(_QWORD **)(a8 + 32);
  }
  sub_D4BD20(v22, a8, v11, v12, v13, v14);
  sub_9C6650(&v19);
  v19 = v22[0];
  if ( v22[0] )
  {
    sub_B976B0((__int64)v22, v22[0], (__int64)&v19);
    v22[0] = 0;
  }
  sub_9C6650(v22);
LABEL_7:
  sub_B157E0((__int64)&v20, &v19);
  sub_B17850((__int64)v22, v18, a5, a6, &v20, v15);
  sub_B18290((__int64)v22, "loop not vectorized: ", 0x15u);
  sub_B18290((__int64)v22, a3, a4);
  sub_1049740(a7, (__int64)v22);
  v22[0] = (unsigned __int8 *)&unk_49D9D40;
  sub_23FD590((__int64)v23);
  if ( v19 )
    sub_B91220((__int64)&v19, (__int64)v19);
}
