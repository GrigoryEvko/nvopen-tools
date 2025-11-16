// Function: sub_337F220
// Address: 0x337f220
//
void __fastcall sub_337F220(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 **a5)
{
  __int64 v5; // r12
  __int64 v7; // rax
  int v8; // edx
  int v9; // eax
  int v10; // r13d
  int v11; // r9d
  __m128i v12; // xmm0
  __m128i v13; // xmm1
  __int64 *v14; // rax
  char v15; // al
  _QWORD *v16; // rcx
  __int64 v17; // r11
  __int64 v18; // r10
  __int64 v19; // rax
  _QWORD *v20; // rdi
  int v21; // eax
  unsigned __int64 v22; // rax
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r15
  int v26; // edx
  int v27; // r13d
  _QWORD *v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 *v31; // rax
  unsigned __int16 v32; // r15
  __int64 v33; // r8
  bool v34; // al
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // rdx
  unsigned __int16 v38; // ax
  char v39; // [rsp+0h] [rbp-F0h]
  __int64 v40; // [rsp+0h] [rbp-F0h]
  int v41; // [rsp+8h] [rbp-E8h]
  int v42; // [rsp+8h] [rbp-E8h]
  __int64 v44; // [rsp+18h] [rbp-D8h]
  int v45; // [rsp+18h] [rbp-D8h]
  __int64 v46; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v47; // [rsp+38h] [rbp-B8h]
  __int64 v48; // [rsp+40h] [rbp-B0h] BYREF
  int v49; // [rsp+48h] [rbp-A8h]
  __int64 v50[2]; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v51; // [rsp+60h] [rbp-90h]
  __m128i v52; // [rsp+70h] [rbp-80h] BYREF
  __m128i v53; // [rsp+80h] [rbp-70h] BYREF
  _QWORD v54[2]; // [rsp+90h] [rbp-60h] BYREF
  __m128i v55; // [rsp+A0h] [rbp-50h]
  __m128i v56; // [rsp+B0h] [rbp-40h]

  v5 = a2;
  v46 = a3;
  v7 = *a1;
  v8 = *((_DWORD *)a1 + 212);
  v47 = a4;
  v48 = 0;
  v49 = v8;
  if ( v7 )
  {
    if ( &v48 != (__int64 *)(v7 + 48) )
    {
      a2 = *(_QWORD *)(v7 + 48);
      v48 = a2;
      if ( a2 )
        sub_B96E90((__int64)&v48, a2, 1);
    }
  }
  v44 = *(_QWORD *)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF));
  LOWORD(v9) = sub_B5A5E0(v5);
  v10 = v9;
  if ( !BYTE1(v9) )
  {
    v32 = v46;
    v33 = a1[108];
    if ( (_WORD)v46 )
    {
      if ( (unsigned __int16)(v46 - 17) <= 0xD3u )
      {
        v37 = 0;
        v32 = word_4456580[(unsigned __int16)v46 - 1];
        goto LABEL_27;
      }
    }
    else
    {
      v40 = a1[108];
      v34 = sub_30070B0((__int64)&v46);
      v33 = v40;
      if ( v34 )
      {
        v38 = sub_3009970((__int64)&v46, a2, v35, v36, v40);
        v33 = v40;
        v32 = v38;
        goto LABEL_27;
      }
    }
    v37 = v47;
LABEL_27:
    LOBYTE(v10) = sub_33CC4A0(v33, v32, v37);
  }
  sub_B91FC0(v52.m128i_i64, v5);
  if ( (*(_BYTE *)(v5 + 7) & 0x20) != 0 && sub_B91C10(v5, 29) && (*(_BYTE *)(v5 + 7) & 0x20) != 0 )
    v11 = sub_B91C10(v5, 4);
  else
    v11 = 0;
  v12 = _mm_load_si128(&v52);
  v13 = _mm_load_si128(&v53);
  v54[0] = v44;
  v54[1] = 0xBFFFFFFFFFFFFFFELL;
  v14 = (__int64 *)a1[109];
  v55 = v12;
  v56 = v13;
  if ( v14 && (v41 = v11, v15 = sub_CF4FA0(*v14, (__int64)v54, (__int64)(v14 + 1), 0), v11 = v41, !v15) )
  {
    v16 = (_QWORD *)a1[108];
    v39 = 0;
    LODWORD(v18) = 0;
    LODWORD(v17) = (_DWORD)v16 + 288;
  }
  else
  {
    v16 = (_QWORD *)a1[108];
    v39 = 1;
    v17 = v16[48];
    v18 = v16[49];
  }
  v19 = *(_QWORD *)(v44 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v19 + 8) - 17 <= 1 )
    v19 = **(_QWORD **)(v19 + 16);
  v20 = (_QWORD *)v16[5];
  v21 = *(_DWORD *)(v19 + 8) >> 8;
  v50[1] = 0;
  LODWORD(v51) = v21;
  BYTE4(v51) = 0;
  v42 = v18;
  v45 = v17;
  v50[0] = 0;
  v22 = sub_2E7BD70(v20, 1u, -1, v10, (int)&v52, v11, 0, v51, 1u, 0, 0);
  v25 = sub_33F1CD0(
          a1[108],
          v46,
          v47,
          (unsigned int)&v48,
          v45,
          v42,
          **a5,
          (*a5)[1],
          (*a5)[2],
          (*a5)[3],
          (*a5)[4],
          (*a5)[5],
          (*a5)[6],
          (*a5)[7],
          v22,
          0);
  v27 = v26;
  if ( v39 )
  {
    v30 = *((unsigned int *)a1 + 34);
    if ( v30 + 1 > (unsigned __int64)*((unsigned int *)a1 + 35) )
    {
      sub_C8D5F0((__int64)(a1 + 16), a1 + 18, v30 + 1, 0x10u, v23, v24);
      v30 = *((unsigned int *)a1 + 34);
    }
    v31 = (__int64 *)(a1[16] + 16 * v30);
    *v31 = v25;
    v31[1] = 1;
    ++*((_DWORD *)a1 + 34);
  }
  v50[0] = v5;
  v28 = sub_337DC20((__int64)(a1 + 1), v50);
  *v28 = v25;
  v29 = v48;
  *((_DWORD *)v28 + 2) = v27;
  if ( v29 )
    sub_B91220((__int64)&v48, v29);
}
