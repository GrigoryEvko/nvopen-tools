// Function: sub_3796360
// Address: 0x3796360
//
unsigned __int8 *__fastcall sub_3796360(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // ebx
  unsigned int v6; // r14d
  unsigned __int16 *v8; // rdx
  int v9; // eax
  __int64 v10; // rdx
  unsigned int *v11; // rax
  __int64 v12; // rsi
  __m128i v13; // xmm0
  __int64 v14; // rax
  unsigned __int16 v15; // dx
  __int64 v16; // r8
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r8
  int v22; // eax
  __int128 v23; // rax
  __int64 v24; // r9
  unsigned __int8 *v25; // r8
  unsigned int v26; // edx
  __int64 v27; // rcx
  __int64 v28; // rsi
  __int64 v29; // r14
  int v30; // r9d
  __int64 v31; // rsi
  unsigned __int8 *v32; // r14
  __int64 v34; // rdx
  unsigned int v35; // edx
  __int64 v36; // rdx
  __int64 v37; // [rsp+0h] [rbp-A0h]
  int v38; // [rsp+0h] [rbp-A0h]
  unsigned int v39; // [rsp+8h] [rbp-98h]
  _QWORD *v40; // [rsp+10h] [rbp-90h]
  unsigned __int8 *v41; // [rsp+10h] [rbp-90h]
  __int64 v42; // [rsp+18h] [rbp-88h]
  unsigned __int16 v43; // [rsp+30h] [rbp-70h] BYREF
  __int64 v44; // [rsp+38h] [rbp-68h]
  __int64 v45; // [rsp+40h] [rbp-60h] BYREF
  int v46; // [rsp+48h] [rbp-58h]
  __int64 v47; // [rsp+50h] [rbp-50h] BYREF
  __int64 v48; // [rsp+58h] [rbp-48h]

  v8 = *(unsigned __int16 **)(a2 + 48);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  LOWORD(v47) = v9;
  v48 = v10;
  if ( (_WORD)v9 )
  {
    v42 = 0;
    LOWORD(v9) = word_4456580[v9 - 1];
  }
  else
  {
    v9 = sub_3009970((__int64)&v47, a2, v10, a4, a5);
    v42 = v36;
    HIWORD(v5) = HIWORD(v9);
  }
  LOWORD(v5) = v9;
  v11 = *(unsigned int **)(a2 + 40);
  v12 = *(_QWORD *)(a2 + 80);
  v13 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v14 = *(_QWORD *)(*(_QWORD *)v11 + 48LL) + 16LL * v11[2];
  v15 = *(_WORD *)v14;
  v16 = *(_QWORD *)(v14 + 8);
  v45 = v12;
  v43 = v15;
  v44 = v16;
  if ( v12 )
  {
    sub_B96E90((__int64)&v45, v12, 1);
    v15 = v43;
    v16 = v44;
  }
  v17 = *a1;
  v46 = *(_DWORD *)(a2 + 72);
  sub_2FE6CC0((__int64)&v47, v17, *(_QWORD *)(a1[1] + 64), v15, v16);
  if ( (_BYTE)v47 == 5 )
  {
    v25 = (unsigned __int8 *)sub_37946F0((__int64)a1, v13.m128i_u64[0], v13.m128i_i64[1]);
    v27 = v35;
  }
  else
  {
    if ( v43 )
    {
      v21 = 0;
      LOWORD(v22) = word_4456580[v43 - 1];
    }
    else
    {
      v22 = sub_3009970((__int64)&v43, v17, v18, v19, v20);
      HIWORD(v6) = HIWORD(v22);
      v21 = v34;
    }
    v37 = v21;
    LOWORD(v6) = v22;
    v40 = (_QWORD *)a1[1];
    *(_QWORD *)&v23 = sub_3400EE0((__int64)v40, 0, (__int64)&v45, 0, v13);
    v25 = sub_3406EB0(v40, 0x9Eu, (__int64)&v45, v6, v37, v24, *(_OWORD *)&v13, v23);
    v27 = v26;
  }
  v28 = *(_QWORD *)(a2 + 80);
  v29 = a1[1];
  v30 = *(_DWORD *)(a2 + 28);
  v47 = v28;
  if ( v28 )
  {
    v38 = v30;
    v39 = v27;
    v41 = v25;
    sub_B96E90((__int64)&v47, v28, 1);
    v30 = v38;
    v27 = v39;
    v25 = v41;
  }
  v31 = *(unsigned int *)(a2 + 24);
  LODWORD(v48) = *(_DWORD *)(a2 + 72);
  v32 = sub_33FA050(v29, v31, (__int64)&v47, v5, v42, v30, v25, v27 | v13.m128i_i64[1] & 0xFFFFFFFF00000000LL);
  if ( v47 )
    sub_B91220((__int64)&v47, v47);
  if ( v45 )
    sub_B91220((__int64)&v45, v45);
  return v32;
}
