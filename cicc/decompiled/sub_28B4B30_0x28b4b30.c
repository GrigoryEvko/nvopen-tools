// Function: sub_28B4B30
// Address: 0x28b4b30
//
_BOOL8 __fastcall sub_28B4B30(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  _QWORD *v4; // r15
  __int64 *v5; // rax
  __m128i v6; // xmm0
  __m128i v7; // xmm1
  __m128i v8; // xmm2
  unsigned __int8 *v9; // rsi
  __int64 *v10; // rax
  __int64 *v11; // rax
  char v12; // bl
  _BOOL8 result; // rax
  bool v14; // r8
  __m128i v15; // [rsp+10h] [rbp-340h] BYREF
  __m128i v16; // [rsp+20h] [rbp-330h] BYREF
  __m128i v17; // [rsp+30h] [rbp-320h] BYREF
  __m128i v18[3]; // [rsp+40h] [rbp-310h] BYREF
  char v19; // [rsp+70h] [rbp-2E0h]
  _QWORD v20[2]; // [rsp+80h] [rbp-2D0h] BYREF
  __int64 v21; // [rsp+90h] [rbp-2C0h]
  __int64 v22; // [rsp+98h] [rbp-2B8h] BYREF
  unsigned int v23; // [rsp+A0h] [rbp-2B0h]
  _QWORD v24[2]; // [rsp+1D8h] [rbp-178h] BYREF
  char v25; // [rsp+1E8h] [rbp-168h]
  _BYTE *v26; // [rsp+1F0h] [rbp-160h]
  __int64 v27; // [rsp+1F8h] [rbp-158h]
  _BYTE v28[128]; // [rsp+200h] [rbp-150h] BYREF
  __int16 v29; // [rsp+280h] [rbp-D0h]
  _QWORD v30[2]; // [rsp+288h] [rbp-C8h] BYREF
  __int64 v31; // [rsp+298h] [rbp-B8h]
  __int64 v32; // [rsp+2A0h] [rbp-B0h] BYREF
  unsigned int v33; // [rsp+2A8h] [rbp-A8h]
  char v34; // [rsp+320h] [rbp-30h] BYREF

  v3 = **(_QWORD **)a1;
  if ( *(_QWORD *)(v3 + 40) != *(_QWORD *)(a2 + 40) || (v14 = sub_B445A0(v3, a2), result = 0, !v14) )
  {
    v4 = *(_QWORD **)(a1 + 8);
    sub_D665A0(&v15, a2);
    v5 = *(__int64 **)a1;
    v6 = _mm_loadu_si128(&v15);
    v19 = 1;
    v7 = _mm_loadu_si128(&v16);
    v8 = _mm_loadu_si128(&v17);
    v18[0] = v6;
    v18[1] = v7;
    v18[2] = v8;
    v9 = (unsigned __int8 *)*v5;
    v10 = &v22;
    v20[0] = v4;
    v20[1] = 0;
    v21 = 1;
    do
    {
      *v10 = -4;
      v10 += 5;
      *(v10 - 4) = -3;
      *(v10 - 3) = -4;
      *(v10 - 2) = -3;
    }
    while ( v10 != v24 );
    v27 = 0x400000000LL;
    v24[0] = v30;
    v24[1] = 0;
    v25 = 0;
    v26 = v28;
    v30[1] = 0;
    v31 = 1;
    v29 = 256;
    v30[0] = &unk_49DDBE8;
    v11 = &v32;
    do
    {
      *v11 = -4096;
      v11 += 2;
    }
    while ( v11 != (__int64 *)&v34 );
    v12 = sub_CF63E0(v4, v9, v18, (__int64)v20);
    v30[0] = &unk_49DDBE8;
    if ( (v31 & 1) == 0 )
      sub_C7D6A0(v32, 16LL * v33, 8);
    nullsub_184();
    if ( v26 != v28 )
      _libc_free((unsigned __int64)v26);
    if ( (v21 & 1) == 0 )
      sub_C7D6A0(v22, 40LL * v23, 8);
    return (v12 & 2) != 0;
  }
  return result;
}
