// Function: sub_337EB70
// Address: 0x337eb70
//
void __fastcall sub_337EB70(__int64 *a1, __int64 a2, unsigned int a3, __int64 a4, __int64 **a5)
{
  __int64 v7; // rax
  int v8; // edx
  __int64 v9; // rsi
  __int64 v10; // r12
  int v11; // r9d
  __m128i v12; // xmm0
  __m128i v13; // xmm1
  __int64 *v14; // rax
  char v15; // al
  _QWORD *v16; // rax
  __int64 v17; // r11
  __int64 v18; // r10
  _QWORD *v19; // rdi
  int v20; // eax
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r12
  int v26; // edx
  int v27; // ebx
  _QWORD *v28; // rax
  __int64 v29; // rsi
  char v30; // al
  __int64 v31; // rax
  __int64 *v32; // rax
  int v33; // [rsp+0h] [rbp-E0h]
  int v35; // [rsp+10h] [rbp-D0h]
  int v36; // [rsp+10h] [rbp-D0h]
  int v37; // [rsp+10h] [rbp-D0h]
  char v40; // [rsp+2Dh] [rbp-B3h]
  __int16 v41; // [rsp+2Eh] [rbp-B2h]
  __int64 v42; // [rsp+30h] [rbp-B0h] BYREF
  int v43; // [rsp+38h] [rbp-A8h]
  __int128 v44; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v45; // [rsp+50h] [rbp-90h]
  __m128i v46; // [rsp+60h] [rbp-80h] BYREF
  __m128i v47; // [rsp+70h] [rbp-70h] BYREF
  _QWORD v48[2]; // [rsp+80h] [rbp-60h] BYREF
  __m128i v49; // [rsp+90h] [rbp-50h]
  __m128i v50; // [rsp+A0h] [rbp-40h]

  v7 = *a1;
  v8 = *((_DWORD *)a1 + 212);
  v42 = 0;
  v43 = v8;
  if ( v7 )
  {
    if ( &v42 != (__int64 *)(v7 + 48) )
    {
      v9 = *(_QWORD *)(v7 + 48);
      v42 = v9;
      if ( v9 )
        sub_B96E90((__int64)&v42, v9, 1);
    }
  }
  v10 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v41 = sub_B5A5E0(a2);
  sub_B91FC0(v46.m128i_i64, a2);
  if ( (*(_BYTE *)(a2 + 7) & 0x20) == 0 || !sub_B91C10(a2, 29) || (*(_BYTE *)(a2 + 7) & 0x20) == 0 )
  {
    v11 = 0;
    if ( HIBYTE(v41) )
      goto LABEL_9;
LABEL_21:
    v37 = v11;
    v30 = sub_33CC4A0(a1[108], a3, a4);
    v11 = v37;
    LOBYTE(v41) = v30;
    goto LABEL_9;
  }
  v11 = sub_B91C10(a2, 4);
  if ( !HIBYTE(v41) )
    goto LABEL_21;
LABEL_9:
  v12 = _mm_load_si128(&v46);
  v13 = _mm_load_si128(&v47);
  v48[0] = v10;
  v48[1] = 0xBFFFFFFFFFFFFFFELL;
  v14 = (__int64 *)a1[109];
  v49 = v12;
  v50 = v13;
  if ( v14 && (v35 = v11, v15 = sub_CF4FA0(*v14, (__int64)v48, (__int64)(v14 + 1), 0), v11 = v35, !v15) )
  {
    v16 = (_QWORD *)a1[108];
    v40 = 0;
    LODWORD(v18) = 0;
    LODWORD(v17) = (_DWORD)v16 + 288;
  }
  else
  {
    v16 = (_QWORD *)a1[108];
    v40 = 1;
    v17 = v16[48];
    v18 = v16[49];
  }
  v19 = (_QWORD *)v16[5];
  BYTE4(v45) = 0;
  *((_QWORD *)&v44 + 1) = 0;
  *(_QWORD *)&v44 = v10 & 0xFFFFFFFFFFFFFFFBLL;
  v20 = 0;
  if ( v10 )
  {
    v21 = *(_QWORD *)(v10 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v21 + 8) - 17 <= 1 )
      v21 = **(_QWORD **)(v21 + 16);
    v20 = *(_DWORD *)(v21 + 8) >> 8;
  }
  LODWORD(v45) = v20;
  v33 = v18;
  v36 = v17;
  v22 = sub_2E7BD70(v19, 1u, -1, (unsigned __int8)v41, (int)&v46, v11, v44, v45, 1u, 0, 0);
  v25 = sub_33F1C00(
          a1[108],
          a3,
          a4,
          (unsigned int)&v42,
          v36,
          v33,
          **a5,
          (*a5)[1],
          (*a5)[2],
          (*a5)[3],
          (*a5)[4],
          (*a5)[5],
          v22,
          0);
  v27 = v26;
  if ( v40 )
  {
    v31 = *((unsigned int *)a1 + 34);
    if ( v31 + 1 > (unsigned __int64)*((unsigned int *)a1 + 35) )
    {
      sub_C8D5F0((__int64)(a1 + 16), a1 + 18, v31 + 1, 0x10u, v23, v24);
      v31 = *((unsigned int *)a1 + 34);
    }
    v32 = (__int64 *)(a1[16] + 16 * v31);
    *v32 = v25;
    v32[1] = 1;
    ++*((_DWORD *)a1 + 34);
  }
  *(_QWORD *)&v44 = a2;
  v28 = sub_337DC20((__int64)(a1 + 1), (__int64 *)&v44);
  *v28 = v25;
  v29 = v42;
  *((_DWORD *)v28 + 2) = v27;
  if ( v29 )
    sub_B91220((__int64)&v42, v29);
}
