// Function: sub_27FEC80
// Address: 0x27fec80
//
__int64 __fastcall sub_27FEC80(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // r14
  __m128i *v11; // rdx
  __m128i si128; // xmm0
  __m128i v13; // xmm0
  const char *v14; // rax
  size_t v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  _BYTE *v19; // rdi
  unsigned __int8 *v20; // rsi
  unsigned __int64 v21; // rax
  __int64 *v22; // rax
  int v23; // eax
  _BYTE *v24; // rsi
  __int64 v25; // r15
  __int64 *v26; // r8
  int v27; // edi
  unsigned int v28; // ecx
  __int64 *v29; // rdx
  __int64 v30; // r11
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r14
  const char *v34; // rax
  size_t v35; // rdx
  _WORD *v36; // rdi
  unsigned __int8 *v37; // rsi
  unsigned __int64 v38; // rax
  _QWORD *v39; // rax
  _BYTE *v40; // rdi
  __int64 v42; // rax
  __int64 v43; // rax
  int v44; // edx
  int v45; // r10d
  unsigned __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // [rsp+10h] [rbp-C0h]
  size_t v49; // [rsp+10h] [rbp-C0h]
  size_t v50; // [rsp+18h] [rbp-B8h]
  __int64 v51; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v52; // [rsp+28h] [rbp-A8h]
  __int64 *v53; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int v54; // [rsp+38h] [rbp-98h]
  _BYTE *v55; // [rsp+70h] [rbp-60h] BYREF
  __int64 v56; // [rsp+78h] [rbp-58h]
  _BYTE v57[80]; // [rsp+80h] [rbp-50h] BYREF

  v8 = sub_BC1CD0(a4, &unk_4F86D28, a3) + 8;
  v9 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  v10 = *a2;
  v48 = v9 + 8;
  v11 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v11 <= 0x36u )
  {
    v10 = sub_CB6200(v10, "Printing analysis 'Loop Access Analysis' for function '", 0x37u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F70AE0);
    v11[3].m128i_i32[0] = 1869182051;
    v11[3].m128i_i16[2] = 8302;
    *v11 = si128;
    v13 = _mm_load_si128((const __m128i *)&xmmword_4394AC0);
    v11[3].m128i_i8[6] = 39;
    v11[1] = v13;
    v11[2] = _mm_load_si128((const __m128i *)&xmmword_4394AD0);
    *(_QWORD *)(v10 + 32) += 55LL;
  }
  v14 = sub_BD5D20(a3);
  v19 = *(_BYTE **)(v10 + 32);
  v20 = (unsigned __int8 *)v14;
  v21 = *(_QWORD *)(v10 + 24) - (_QWORD)v19;
  if ( v21 < v15 )
  {
    v47 = sub_CB6200(v10, v20, v15);
    v19 = *(_BYTE **)(v47 + 32);
    v10 = v47;
    v21 = *(_QWORD *)(v47 + 24) - (_QWORD)v19;
  }
  else if ( v15 )
  {
    v50 = v15;
    memcpy(v19, v20, v15);
    v19 = (_BYTE *)(v50 + *(_QWORD *)(v10 + 32));
    v46 = *(_QWORD *)(v10 + 24) - (_QWORD)v19;
    *(_QWORD *)(v10 + 32) = v19;
    if ( v46 > 2 )
      goto LABEL_6;
LABEL_38:
    sub_CB6200(v10, "':\n", 3u);
    goto LABEL_7;
  }
  if ( v21 <= 2 )
    goto LABEL_38;
LABEL_6:
  v19[2] = 10;
  *(_WORD *)v19 = 14887;
  *(_QWORD *)(v10 + 32) += 3LL;
LABEL_7:
  v22 = (__int64 *)&v53;
  v51 = 0;
  v52 = 1;
  do
  {
    *v22 = -4096;
    v22 += 2;
  }
  while ( v22 != (__int64 *)&v55 );
  v55 = v57;
  v56 = 0x400000000LL;
  sub_F774D0(v48, (__int64)&v51, (__int64)&v55, v16, v17, v18);
  v23 = v56;
  if ( (_DWORD)v56 )
  {
    while ( 1 )
    {
      v24 = v55;
      v25 = *(_QWORD *)&v55[8 * v23 - 8];
      if ( (v52 & 1) != 0 )
        break;
      v26 = v53;
      if ( v54 )
      {
        v27 = v54 - 1;
LABEL_12:
        v28 = v27 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
        v29 = &v26[2 * v28];
        v30 = *v29;
        if ( v25 == *v29 )
        {
LABEL_13:
          *v29 = -8192;
          ++HIDWORD(v52);
          v24 = v55;
          LODWORD(v52) = (2 * ((unsigned int)v52 >> 1) - 2) | v52 & 1;
          v23 = v56;
        }
        else
        {
          v44 = 1;
          while ( v30 != -4096 )
          {
            v45 = v44 + 1;
            v28 = v27 & (v44 + v28);
            v29 = &v26[2 * v28];
            v30 = *v29;
            if ( v25 == *v29 )
              goto LABEL_13;
            v44 = v45;
          }
        }
      }
      v31 = (unsigned int)(v23 - 1);
      v32 = (__int64)&v24[8 * v31 - 8];
      while ( 1 )
      {
        LODWORD(v56) = v31;
        if ( !(_DWORD)v31 )
          break;
        v32 -= 8;
        if ( *(_QWORD *)(v32 + 8) )
          break;
        LODWORD(v31) = v31 - 1;
      }
      v33 = sub_CB69B0(*a2, 2u);
      v34 = sub_BD5D20(**(_QWORD **)(v25 + 32));
      v36 = *(_WORD **)(v33 + 32);
      v37 = (unsigned __int8 *)v34;
      v38 = *(_QWORD *)(v33 + 24) - (_QWORD)v36;
      if ( v38 < v35 )
      {
        v42 = sub_CB6200(v33, v37, v35);
        v36 = *(_WORD **)(v42 + 32);
        v33 = v42;
        v38 = *(_QWORD *)(v42 + 24) - (_QWORD)v36;
      }
      else if ( v35 )
      {
        v49 = v35;
        memcpy(v36, v37, v35);
        v43 = *(_QWORD *)(v33 + 24);
        v36 = (_WORD *)(v49 + *(_QWORD *)(v33 + 32));
        *(_QWORD *)(v33 + 32) = v36;
        v38 = v43 - (_QWORD)v36;
      }
      if ( v38 <= 1 )
      {
        sub_CB6200(v33, (unsigned __int8 *)":\n", 2u);
      }
      else
      {
        *v36 = 2618;
        *(_QWORD *)(v33 + 32) += 2LL;
      }
      v39 = (_QWORD *)sub_D440B0(v8, v25);
      sub_D36E30(v39, *a2, 4u);
      v23 = v56;
      if ( !(_DWORD)v56 )
        goto LABEL_24;
    }
    v26 = (__int64 *)&v53;
    v27 = 3;
    goto LABEL_12;
  }
LABEL_24:
  *(_BYTE *)(a1 + 76) = 1;
  v40 = v55;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)a1 = 1;
  if ( v40 != v57 )
    _libc_free((unsigned __int64)v40);
  if ( (v52 & 1) == 0 )
    sub_C7D6A0((__int64)v53, 16LL * v54, 8);
  return a1;
}
