// Function: sub_36EA5D0
// Address: 0x36ea5d0
//
void __fastcall sub_36EA5D0(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // rsi
  __int64 v7; // rax
  _QWORD *v8; // r13
  unsigned int v9; // r15d
  int v10; // r13d
  _BYTE *v11; // rax
  unsigned int v13; // eax
  int v14; // ecx
  __int64 v15; // r9
  int v16; // r14d
  __int64 v17; // rax
  __m128i v18; // xmm0
  __int64 v19; // rsi
  __int64 v20; // rdi
  __int32 v21; // edx
  _QWORD *v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // r13
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // [rsp+0h] [rbp-90h]
  __int64 v29; // [rsp+8h] [rbp-88h]
  __int64 v30; // [rsp+10h] [rbp-80h] BYREF
  int v31; // [rsp+18h] [rbp-78h]
  __int64 v32; // [rsp+20h] [rbp-70h] BYREF
  int v33; // [rsp+28h] [rbp-68h]
  __m128i v34; // [rsp+30h] [rbp-60h] BYREF
  __m128i v35; // [rsp+40h] [rbp-50h]
  __m128i v36; // [rsp+50h] [rbp-40h]

  if ( *(_DWORD *)(*(_QWORD *)(a1 + 1136) + 344LL) <= 0x48u )
    goto LABEL_42;
  v4 = *(_QWORD *)(a2 + 80);
  v30 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v30, v4, 1);
  v31 = *(_DWORD *)(a2 + 72);
  v7 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL) + 96LL);
  v8 = *(_QWORD **)(v7 + 24);
  if ( *(_DWORD *)(v7 + 32) > 0x40u )
    v8 = (_QWORD *)*v8;
  v9 = (unsigned int)v8;
  v10 = (unsigned __int8)v8 & 0x1E;
  if ( v10 )
  {
    v11 = sub_C94E20((__int64)qword_4F86410);
    if ( !(v11 ? *v11 : LOBYTE(qword_4F86410[2])) )
LABEL_42:
      sub_C64ED0("Not supported on this architecture", 1u);
  }
  v28 = *(_QWORD *)(a2 + 112);
  v29 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 64) + 40LL));
  v13 = sub_2EAC1E0(v28);
  v14 = sub_AE2980(v29, v13)[1];
  switch ( (v9 >> 1) & 0xF )
  {
    case 0u:
      if ( a3 == 1 )
      {
        v16 = (v14 != 64) + 2653;
      }
      else if ( a3 == 2 )
      {
        v16 = (v14 != 64) + 2655;
      }
      else
      {
        v16 = (v14 != 64) + 2657;
      }
      break;
    case 1u:
      if ( a3 == 1 )
      {
        v16 = (v14 != 64) + 2635;
      }
      else if ( a3 == 2 )
      {
        v16 = (v14 != 64) + 2637;
      }
      else
      {
        v16 = (v14 != 64) + 2639;
      }
      break;
    case 2u:
      if ( a3 == 1 )
      {
        v16 = (v14 != 64) + 2629;
      }
      else if ( a3 == 2 )
      {
        v16 = (v14 != 64) + 2631;
      }
      else
      {
        v16 = (v14 != 64) + 2633;
      }
      break;
    case 3u:
      if ( a3 == 1 )
      {
        v16 = (v14 != 64) + 2647;
      }
      else if ( a3 == 2 )
      {
        v16 = (v14 != 64) + 2649;
      }
      else
      {
        v16 = (v14 != 64) + 2651;
      }
      break;
    case 4u:
      if ( a3 == 1 )
      {
        v16 = (v14 != 64) + 2641;
      }
      else if ( a3 == 2 )
      {
        v16 = (v14 != 64) + 2643;
      }
      else
      {
        v16 = (v14 != 64) + 2645;
      }
      break;
    default:
      sub_C64ED0("Unknown size", 1u);
  }
  v17 = *(_QWORD *)(a2 + 40);
  if ( (_BYTE)v10 )
  {
    v22 = *(_QWORD **)(a1 + 64);
    v23 = 2;
    v34 = _mm_loadu_si128((const __m128i *)(v17 + 120));
    v35 = _mm_loadu_si128((const __m128i *)v17);
  }
  else
  {
    v18 = _mm_loadu_si128((const __m128i *)(v17 + 120));
    v19 = *(_QWORD *)(a2 + 80);
    v32 = v19;
    v34 = v18;
    if ( v19 )
      sub_B96E90((__int64)&v32, v19, 1);
    v20 = *(_QWORD *)(a1 + 64);
    v33 = *(_DWORD *)(a2 + 72);
    v35.m128i_i64[0] = (__int64)sub_3400BD0(v20, v9 & 1, (__int64)&v32, 7, 0, 1u, v18, 0);
    v35.m128i_i32[2] = v21;
    if ( v32 )
      sub_B91220((__int64)&v32, v32);
    v22 = *(_QWORD **)(a1 + 64);
    v23 = 3;
    v36 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  }
  v24 = sub_33E66D0(
          v22,
          v16,
          (__int64)&v30,
          *(_QWORD *)(a2 + 48),
          *(unsigned int *)(a2 + 68),
          v15,
          (unsigned __int64 *)&v34,
          v23);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v24, v25, v26, v27);
  sub_3421DB0(v24);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v30 )
    sub_B91220((__int64)&v30, v30);
}
