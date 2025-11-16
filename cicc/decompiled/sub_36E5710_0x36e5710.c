// Function: sub_36E5710
// Address: 0x36e5710
//
void __fastcall sub_36E5710(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rsi
  int v6; // edi
  __int64 v7; // rdx
  __int64 v8; // rax
  bool v9; // cc
  _QWORD *v10; // rax
  __int64 v11; // r8
  _QWORD *v12; // rcx
  unsigned __int8 v13; // cl
  int v14; // r15d
  __m128i v15; // xmm0
  __m128i v16; // xmm1
  __int32 v17; // edx
  __int64 v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // r9
  int v21; // edx
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // r12
  __m128i v27; // xmm4
  int v28; // edx
  __int64 v29; // rsi
  __int64 v30; // rdi
  __int64 v31; // r9
  int v32; // edx
  unsigned __int8 v33; // [rsp+7h] [rbp-A9h]
  unsigned __int8 v34; // [rsp+7h] [rbp-A9h]
  __int64 v35; // [rsp+8h] [rbp-A8h]
  __int64 v36; // [rsp+10h] [rbp-A0h] BYREF
  int v37; // [rsp+18h] [rbp-98h]
  __int64 v38; // [rsp+20h] [rbp-90h] BYREF
  int v39; // [rsp+28h] [rbp-88h]
  __m128i v40; // [rsp+30h] [rbp-80h] BYREF
  __m128i v41; // [rsp+40h] [rbp-70h]
  __m128i v42; // [rsp+50h] [rbp-60h]
  unsigned __int8 *v43; // [rsp+60h] [rbp-50h]
  int v44; // [rsp+68h] [rbp-48h]
  unsigned __int8 *v45; // [rsp+70h] [rbp-40h]
  int v46; // [rsp+78h] [rbp-38h]

  if ( *(_DWORD *)(*(_QWORD *)(a1 + 1136) + 344LL) <= 0x63u )
    sub_C64ED0("F32x2 intrinsics are not supported on this architecture", 1u);
  v4 = *(_QWORD *)(a2 + 80);
  v36 = v4;
  if ( v4 )
  {
    sub_B96E90((__int64)&v36, v4, 1);
    v4 = *(_QWORD *)(a2 + 80);
  }
  v6 = *(_DWORD *)(a2 + 72);
  v7 = *(_QWORD *)(a2 + 40);
  v37 = v6;
  v8 = *(_QWORD *)(*(_QWORD *)v7 + 96LL);
  v9 = *(_DWORD *)(v8 + 32) <= 0x40u;
  v10 = *(_QWORD **)(v8 + 24);
  if ( !v9 )
    v10 = (_QWORD *)*v10;
  v11 = *(_QWORD *)(*(_QWORD *)(v7 + 40) + 96LL);
  v12 = *(_QWORD **)(v11 + 24);
  if ( *(_DWORD *)(v11 + 32) > 0x40u )
    v12 = (_QWORD *)*v12;
  v13 = (unsigned __int8)v12 & 7;
  if ( (_DWORD)v10 == 9157 )
  {
    v35 = 1;
    v14 = 2944;
    goto LABEL_14;
  }
  if ( (unsigned int)v10 > 0x23C5 )
  {
    if ( (_DWORD)v10 == 9579 )
    {
      v35 = 1;
    }
    else
    {
      v35 = 0;
      v14 = 2944;
      if ( (_DWORD)v10 == 9160 )
        goto LABEL_14;
    }
    v14 = (unsigned int)((_DWORD)v10 - 9579) < 2 ? 0xE8E : 0;
    goto LABEL_14;
  }
  if ( (_DWORD)v10 == 8134 )
  {
    v35 = 1;
    v14 = 315;
    goto LABEL_14;
  }
  if ( (_DWORD)v10 == 8650 )
  {
    v35 = 1;
  }
  else
  {
    if ( (unsigned int)v10 <= 0x1FC7 )
    {
      v35 = 0;
      v14 = (unsigned int)v10 >= 0x1FC6 ? 0x13B : 0;
LABEL_14:
      v15 = _mm_loadu_si128((const __m128i *)(v7 + 80));
      v40 = v15;
      v16 = _mm_loadu_si128((const __m128i *)(v7 + 120));
      v38 = v4;
      v41 = v16;
      if ( v4 )
      {
        v33 = v13;
        sub_B96E90((__int64)&v38, v4, 1);
        v6 = *(_DWORD *)(a2 + 72);
        v13 = v33;
      }
      v39 = v6;
      v42.m128i_i64[0] = (__int64)sub_3400BD0(*(_QWORD *)(a1 + 64), v13, (__int64)&v38, 7, 0, 1u, v15, 0);
      v42.m128i_i32[2] = v17;
      if ( v38 )
        sub_B91220((__int64)&v38, v38);
      v18 = *(_QWORD *)(a2 + 80);
      v38 = v18;
      if ( v18 )
        sub_B96E90((__int64)&v38, v18, 1);
      v19 = *(_QWORD *)(a1 + 64);
      v39 = *(_DWORD *)(a2 + 72);
      v43 = sub_3400BD0(v19, v35, (__int64)&v38, 7, 0, 1u, v15, 0);
      v44 = v21;
      if ( v38 )
        sub_B91220((__int64)&v38, v38);
      v22 = sub_33E66D0(
              *(_QWORD **)(a1 + 64),
              v14,
              (__int64)&v36,
              *(_QWORD *)(a2 + 48),
              *(unsigned int *)(a2 + 68),
              v20,
              (unsigned __int64 *)&v40,
              4);
      goto LABEL_23;
    }
    v35 = 0;
    if ( (unsigned int)((_DWORD)v10 - 8650) > 1 )
    {
      v14 = 0;
      goto LABEL_14;
    }
  }
  v40 = _mm_loadu_si128((const __m128i *)(v7 + 80));
  v41 = _mm_loadu_si128((const __m128i *)(v7 + 120));
  v27 = _mm_loadu_si128((const __m128i *)(v7 + 160));
  v38 = v4;
  v42 = v27;
  if ( v4 )
  {
    v34 = v13;
    sub_B96E90((__int64)&v38, v4, 1);
    v6 = *(_DWORD *)(a2 + 72);
    v13 = v34;
  }
  v39 = v6;
  v43 = sub_3400BD0(*(_QWORD *)(a1 + 64), v13, (__int64)&v38, 7, 0, 1u, a3, 0);
  v44 = v28;
  if ( v38 )
    sub_B91220((__int64)&v38, v38);
  v29 = *(_QWORD *)(a2 + 80);
  v38 = v29;
  if ( v29 )
    sub_B96E90((__int64)&v38, v29, 1);
  v30 = *(_QWORD *)(a1 + 64);
  v39 = *(_DWORD *)(a2 + 72);
  v45 = sub_3400BD0(v30, v35, (__int64)&v38, 7, 0, 1u, a3, 0);
  v46 = v32;
  if ( v38 )
    sub_B91220((__int64)&v38, v38);
  v22 = sub_33E66D0(
          *(_QWORD **)(a1 + 64),
          1417,
          (__int64)&v36,
          *(_QWORD *)(a2 + 48),
          *(unsigned int *)(a2 + 68),
          v31,
          (unsigned __int64 *)&v40,
          5);
LABEL_23:
  v26 = v22;
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v22, v23, v24, v25);
  sub_3421DB0(v26);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v36 )
    sub_B91220((__int64)&v36, v36);
}
