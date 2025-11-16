// Function: sub_1B955D0
// Address: 0x1b955d0
//
void __fastcall sub_1B955D0(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // r15
  char *v5; // rsi
  size_t v6; // rdx
  __int64 v7; // r15
  __int64 v8; // [rsp+8h] [rbp-238h] BYREF
  __m128i v9[2]; // [rsp+10h] [rbp-230h] BYREF
  _QWORD v10[11]; // [rsp+30h] [rbp-210h] BYREF
  _BYTE v11[440]; // [rsp+88h] [rbp-1B8h] BYREF

  if ( *(_DWORD *)(a2 + 8) == 1 )
  {
    if ( *(_DWORD *)(a2 + 24) == 1 )
      return;
    v4 = **(_QWORD **)(a1 + 32);
    sub_13FD840(&v8, a1);
    sub_15C9090((__int64)v9, &v8);
    sub_15CA840((__int64)v10, (__int64)"loop-vectorize", (__int64)"FailedRequestedInterleaving", 27, v9, v4);
    sub_15CAB20((__int64)v10, "loop not interleaved: ", 0x16u);
    v5 = "failed explicitly specified loop interleaving";
    v6 = 45;
  }
  else
  {
    v7 = **(_QWORD **)(a1 + 32);
    sub_13FD840(&v8, a1);
    sub_15C9090((__int64)v9, &v8);
    sub_15CA840((__int64)v10, (__int64)"loop-vectorize", (__int64)"FailedRequestedVectorization", 28, v9, v7);
    sub_15CAB20((__int64)v10, "loop not vectorized: ", 0x15u);
    v5 = "failed explicitly specified loop vectorization";
    v6 = 46;
  }
  sub_15CAB20((__int64)v10, v5, v6);
  sub_143AA50(a3, (__int64)v10);
  v10[0] = &unk_49ECF68;
  sub_1897B80((__int64)v11);
  if ( v8 )
    sub_161E7C0((__int64)&v8, v8);
}
