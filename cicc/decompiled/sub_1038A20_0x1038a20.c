// Function: sub_1038A20
// Address: 0x1038a20
//
__int64 __fastcall sub_1038A20(__int64 a1, __int16 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  unsigned int v8; // r12d
  float v10; // xmm0_4
  bool v11; // zf
  __int64 v12; // [rsp-8h] [rbp-40h]
  float v13[9]; // [rsp+14h] [rbp-24h] BYREF

  v7 = a1 + 160;
  v13[0] = 0.0;
  v8 = sub_C55C50(v7, a1, a3, a4, a5, a6, v13);
  if ( (_BYTE)v8 )
    return v8;
  v10 = v13[0];
  v11 = *(_QWORD *)(a1 + 184) == 0;
  *(_WORD *)(a1 + 14) = a2;
  *(float *)(a1 + 136) = v10;
  if ( v11 )
    sub_4263D6(v7, a1, v12);
  (*(void (__fastcall **)(__int64, float *, __int64))(a1 + 192))(a1 + 168, v13, v12);
  return v8;
}
