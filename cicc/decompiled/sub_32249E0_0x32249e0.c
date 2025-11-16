// Function: sub_32249E0
// Address: 0x32249e0
//
void __fastcall sub_32249E0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v6; // rax
  unsigned __int64 v7; // r12
  __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( *(_QWORD *)(a1 + 3072) )
    sub_324AD70(*a4, a3, 27, *(_QWORD *)(a1 + 3064), *(_QWORD *)(a1 + 3072));
  sub_321F6B0(a1, *a4, a3);
  v6 = *a4;
  *a4 = 0;
  v8[0] = v6;
  sub_3245240(a1 + 3776, v8);
  v7 = v8[0];
  if ( v8[0] )
  {
    sub_3223CF0(v8[0]);
    j_j___libc_free_0(v7);
  }
}
