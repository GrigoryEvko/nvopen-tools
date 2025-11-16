// Function: sub_21E3090
// Address: 0x21e3090
//
__int64 __fastcall sub_21E3090(
        __int64 a1,
        __int64 a2,
        __m128i a3,
        double a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  const __m128i *v10; // rdx
  _QWORD *v11; // r14
  __int64 v12; // rcx
  _QWORD *v13; // rax
  __int64 v14; // rsi
  char v15; // r12
  int v16; // ecx
  __int64 v17; // rcx
  int v18; // r8d
  __int64 v19; // r12
  __int64 v21; // rax
  __int64 v22; // r8
  __int64 v23; // rsi
  __int64 v24; // r9
  int v25; // edx
  __int64 v26; // rcx
  int v27; // r8d
  int v28; // [rsp+8h] [rbp-78h]
  __int64 v29; // [rsp+10h] [rbp-70h] BYREF
  int v30; // [rsp+18h] [rbp-68h]
  __int64 v31; // [rsp+20h] [rbp-60h] BYREF
  int v32; // [rsp+28h] [rbp-58h]
  _OWORD v33[5]; // [rsp+30h] [rbp-50h] BYREF

  v10 = *(const __m128i **)(a2 + 32);
  v11 = *(_QWORD **)(a1 - 176);
  v12 = *(_QWORD *)(v10[5].m128i_i64[0] + 88);
  v13 = *(_QWORD **)(v12 + 24);
  if ( *(_DWORD *)(v12 + 32) > 0x40u )
    v13 = (_QWORD *)*v13;
  v14 = *(_QWORD *)(a2 + 72);
  v15 = (char)v13;
  v29 = v14;
  if ( v14 )
  {
    sub_1623A60((__int64)&v29, v14, 2);
    v10 = *(const __m128i **)(a2 + 32);
  }
  v16 = *(_DWORD *)(a2 + 64);
  v30 = v16;
  if ( (v15 & 0xF) == 1 )
  {
    v21 = *(_QWORD *)(v10[7].m128i_i64[1] + 88);
    v22 = *(_QWORD *)(v21 + 24);
    if ( *(_DWORD *)(v21 + 32) > 0x40u )
      v22 = **(_QWORD **)(v21 + 24);
    v23 = *(_QWORD *)(a2 + 72);
    v31 = v23;
    if ( v23 )
    {
      v28 = v22;
      sub_1623A60((__int64)&v31, v23, 2);
      v16 = *(_DWORD *)(a2 + 64);
      LODWORD(v22) = v28;
    }
    v32 = v16;
    *(_QWORD *)&v33[0] = sub_1D38BB0(*(_QWORD *)(a1 - 176), (unsigned int)v22, (__int64)&v31, 5, 0, 1, a3, a4, a5, 0);
    DWORD2(v33[0]) = v25;
    if ( v31 )
      sub_161E7C0((__int64)&v31, v31);
    v26 = *(_QWORD *)(a2 + 40);
    v27 = *(_DWORD *)(a2 + 60);
    v33[1] = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 32));
    v19 = sub_1D23DE0(v11, 3149, (__int64)&v29, v26, v27, v24, (__int64 *)v33, 2);
  }
  else
  {
    v17 = *(_QWORD *)(a2 + 40);
    v18 = *(_DWORD *)(a2 + 60);
    v33[0] = _mm_loadu_si128(v10);
    if ( (v15 & 0xF) == 2 )
      v19 = sub_1D23DE0(v11, 3148, (__int64)&v29, v17, v18, a9, (__int64 *)v33, 1);
    else
      v19 = sub_1D23DE0(v11, 3147, (__int64)&v29, v17, v18, a9, (__int64 *)v33, 1);
  }
  if ( v29 )
    sub_161E7C0((__int64)&v29, v29);
  return v19;
}
