// Function: sub_22550C0
// Address: 0x22550c0
//
void __fastcall sub_22550C0(__int64 a1)
{
  volatile signed __int32 *v1[4]; // [rsp+8h] [rbp-20h] BYREF

  *(_QWORD *)(a1 + 8) = 6;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 4098;
  sub_220A990(v1);
  sub_22090A0((volatile signed __int32 **)(a1 + 208), v1);
  sub_2209150(v1);
}
