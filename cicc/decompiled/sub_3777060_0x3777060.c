// Function: sub_3777060
// Address: 0x3777060
//
__m128i *__fastcall sub_3777060(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r14
  __int64 *v8; // rdi
  __int64 v9; // rcx
  __int16 v10; // ax
  int v11; // eax
  __int16 v12; // cx
  __int64 v13; // rax
  unsigned __int16 *v14; // rax
  __int64 v15; // r8
  unsigned int v16; // ecx
  __int64 v17; // r8
  __int64 v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rcx
  unsigned __int16 *v21; // rdx
  int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // r8
  __int16 v25; // ax
  __m128i *v26; // r14
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r8
  __int64 v31; // rdx
  __int64 v32; // [rsp+8h] [rbp-B8h]
  __int64 v33; // [rsp+8h] [rbp-B8h]
  __int128 v34; // [rsp+10h] [rbp-B0h]
  unsigned __int8 v35; // [rsp+23h] [rbp-9Dh]
  __int16 v36; // [rsp+24h] [rbp-9Ch]
  __int64 *v37; // [rsp+28h] [rbp-98h]
  __int64 v38; // [rsp+30h] [rbp-90h]
  __int64 v39; // [rsp+38h] [rbp-88h]
  __int16 v40; // [rsp+40h] [rbp-80h] BYREF
  __int64 v41; // [rsp+48h] [rbp-78h]
  __int16 v42; // [rsp+50h] [rbp-70h] BYREF
  __int64 v43; // [rsp+58h] [rbp-68h]
  __int64 v44; // [rsp+60h] [rbp-60h] BYREF
  int v45; // [rsp+68h] [rbp-58h]
  _OWORD v46[5]; // [rsp+70h] [rbp-50h] BYREF

  v7 = *(_QWORD *)(a2 + 112);
  v8 = *(__int64 **)(a1 + 8);
  v9 = *(_QWORD *)(a2 + 104);
  v37 = v8;
  v43 = v9;
  v46[1] = _mm_loadu_si128((const __m128i *)(v7 + 56));
  v10 = *(_WORD *)(v7 + 32);
  v46[0] = _mm_loadu_si128((const __m128i *)(v7 + 40));
  v36 = v10;
  v35 = *(_BYTE *)(v7 + 34);
  v11 = *(unsigned __int16 *)(a2 + 96);
  v42 = v11;
  if ( (_WORD)v11 )
  {
    v12 = word_4456580[v11 - 1];
    v13 = 0;
  }
  else
  {
    v28 = sub_3009970((__int64)&v42, a2, a3, v9, a5);
    v8 = *(__int64 **)(a1 + 8);
    v7 = *(_QWORD *)(a2 + 112);
    v30 = v29;
    v12 = v28;
    a3 = v28;
    v13 = v30;
  }
  v39 = v13;
  LOWORD(a3) = v12;
  v38 = a3;
  v14 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 48LL)
                           + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 48LL));
  v15 = *((_QWORD *)v14 + 1);
  v16 = *v14;
  v44 = 0;
  v45 = 0;
  *(_QWORD *)&v34 = sub_33F17F0(v8, 51, (__int64)&v44, v16, v15);
  *((_QWORD *)&v34 + 1) = v18;
  if ( v44 )
    sub_B91220((__int64)&v44, v44);
  v19 = *(_QWORD *)(a2 + 80);
  v20 = *(_QWORD *)(a2 + 40);
  v44 = v19;
  if ( v19 )
  {
    v32 = v20;
    sub_B96E90((__int64)&v44, v19, 1);
    v20 = v32;
  }
  v21 = *(unsigned __int16 **)(a2 + 48);
  v45 = *(_DWORD *)(a2 + 72);
  v22 = *v21;
  v23 = *((_QWORD *)v21 + 1);
  v40 = v22;
  v41 = v23;
  if ( (_WORD)v22 )
  {
    v24 = 0;
    v25 = word_4456580[v22 - 1];
  }
  else
  {
    v33 = v20;
    v25 = sub_3009970((__int64)&v40, v19, v23, v20, v17);
    v20 = v33;
    v24 = v31;
  }
  v26 = sub_33EA290(
          v37,
          0,
          (*(_BYTE *)(a2 + 33) >> 2) & 3,
          v25,
          v24,
          (__int64)&v44,
          *(_OWORD *)v20,
          *(_QWORD *)(v20 + 40),
          *(_QWORD *)(v20 + 48),
          v34,
          *(_OWORD *)v7,
          *(_QWORD *)(v7 + 16),
          v38,
          v39,
          v35,
          v36,
          (__int64)v46,
          0);
  if ( v44 )
    sub_B91220((__int64)&v44, v44);
  sub_3760E70(a1, a2, 1, (unsigned __int64)v26, 1);
  return v26;
}
