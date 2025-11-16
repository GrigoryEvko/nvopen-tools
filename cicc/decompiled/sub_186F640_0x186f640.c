// Function: sub_186F640
// Address: 0x186f640
//
__int64 __fastcall sub_186F640(__int64 a1, char *a2, size_t a3)
{
  char *v5; // rsi
  __int64 v6; // rsi
  size_t v7; // r14
  __int64 v8; // r8
  __int64 *v9; // r9
  __int64 v10; // rax
  unsigned int v11; // r8d
  __int64 *v12; // r9
  __int64 v13; // rcx
  _BYTE *v14; // rdi
  __int64 v15; // rdx
  __int64 v17; // rax
  _BYTE *v18; // rax
  char *v19; // rdi
  _QWORD *v20; // rax
  __m128i *v21; // rdx
  __int64 v22; // r12
  __m128i si128; // xmm0
  __m128i *v24; // rdi
  __int64 v25; // rax
  unsigned __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 *v30; // [rsp+28h] [rbp-288h]
  __int64 *v31; // [rsp+28h] [rbp-288h]
  unsigned int v32; // [rsp+30h] [rbp-280h]
  __int64 *v33; // [rsp+30h] [rbp-280h]
  unsigned int v34; // [rsp+30h] [rbp-280h]
  unsigned int v35; // [rsp+38h] [rbp-278h]
  __int64 v36; // [rsp+38h] [rbp-278h]
  __int64 v37; // [rsp+40h] [rbp-270h]
  unsigned __int8 *srca; // [rsp+48h] [rbp-268h]
  unsigned __int8 *v40; // [rsp+50h] [rbp-260h] BYREF
  size_t n; // [rsp+58h] [rbp-258h]
  _QWORD v42[2]; // [rsp+60h] [rbp-250h] BYREF
  __int64 (__fastcall **v43)(); // [rsp+70h] [rbp-240h] BYREF
  __int64 v44; // [rsp+78h] [rbp-238h]
  _QWORD v45[7]; // [rsp+80h] [rbp-230h] BYREF
  _BYTE v46[48]; // [rsp+B8h] [rbp-1F8h] BYREF
  _BYTE v47[136]; // [rsp+E8h] [rbp-1C8h] BYREF
  _QWORD v48[4]; // [rsp+170h] [rbp-140h] BYREF
  int v49; // [rsp+190h] [rbp-120h]
  __int64 v50; // [rsp+248h] [rbp-68h]
  __int16 v51; // [rsp+250h] [rbp-60h]
  __int64 v52; // [rsp+258h] [rbp-58h]
  __int64 v53; // [rsp+260h] [rbp-50h]
  __int64 v54; // [rsp+268h] [rbp-48h]
  __int64 v55; // [rsp+270h] [rbp-40h]

  sub_222DF20(v48);
  v51 = 0;
  v50 = 0;
  v48[0] = off_4A06798;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v43 = (__int64 (__fastcall **)())qword_4A064D0;
  v55 = 0;
  *(__int64 (__fastcall ***)())((char *)&v43 + qword_4A064D0[-3]) = (__int64 (__fastcall **)())&unk_4A064F8;
  v44 = 0;
  sub_222DD70((char *)&v43 + (_QWORD)*(v43 - 3), 0);
  v43 = off_4A06540;
  v48[0] = off_4A06568;
  sub_222BA80(v45);
  sub_222DD70(v48, v45);
  if ( sub_222C940(v45, a2, 8) )
  {
    v5 = 0;
    sub_222DC80((char *)&v43 + (_QWORD)*(v43 - 3), 0);
  }
  else
  {
    v19 = (char *)&v43 + (_QWORD)*(v43 - 3);
    v5 = (char *)(*((_DWORD *)v19 + 8) | 4u);
    sub_222DC80(v19, v5);
  }
  if ( !v49 )
  {
    while ( 1 )
    {
      v6 = (__int64)&v40;
      v40 = (unsigned __int8 *)v42;
      n = 0;
      LOBYTE(v42[0]) = 0;
      sub_22088A0(&v43, &v40);
      v7 = n;
      if ( n )
      {
        v6 = (__int64)v40;
        srca = v40;
        v8 = (unsigned int)sub_16D19C0(a1, v40, n);
        v9 = (__int64 *)(*(_QWORD *)a1 + 8 * v8);
        if ( !*v9 )
          goto LABEL_12;
        if ( *v9 == -8 )
          break;
      }
LABEL_5:
      if ( v40 != (unsigned __int8 *)v42 )
      {
        v6 = v42[0] + 1LL;
        j_j___libc_free_0(v40, v42[0] + 1LL);
      }
      if ( (v49 & 5) != 0 )
      {
        v43 = off_4A06540;
        v48[0] = off_4A06568;
        v45[0] = off_4A06448;
        sub_222C7F0(v45);
        sub_2207D90(v47);
        v45[0] = off_4A07480;
        sub_2209150(v46, v6, v15);
        v43 = (__int64 (__fastcall **)())qword_4A064D0;
        *(__int64 (__fastcall ***)())((char *)&v43 + qword_4A064D0[-3]) = (__int64 (__fastcall **)())&unk_4A064F8;
        goto LABEL_16;
      }
    }
    --*(_DWORD *)(a1 + 16);
LABEL_12:
    v30 = v9;
    v32 = v8;
    v10 = malloc(v7 + 17);
    v11 = v32;
    v12 = v30;
    v13 = v10;
    if ( !v10 )
    {
      if ( v7 == -17 )
      {
        v17 = malloc(1u);
        v13 = 0;
        v11 = v32;
        v12 = v30;
        if ( v17 )
        {
          v14 = (_BYTE *)(v17 + 16);
          v13 = v17;
          goto LABEL_19;
        }
      }
      v31 = v12;
      v34 = v11;
      v36 = v13;
      sub_16BD1C0("Allocation failed", 1u);
      v13 = v36;
      v11 = v34;
      v12 = v31;
    }
    v14 = (_BYTE *)(v13 + 16);
    if ( v7 + 1 <= 1 )
    {
LABEL_14:
      v14[v7] = 0;
      v6 = v11;
      *(_QWORD *)v13 = v7;
      *(_BYTE *)(v13 + 8) = 0;
      *v12 = v13;
      ++*(_DWORD *)(a1 + 12);
      sub_16D1CD0(a1, v11);
      goto LABEL_5;
    }
LABEL_19:
    v33 = v12;
    v35 = v11;
    v37 = v13;
    v18 = memcpy(v14, srca, v7);
    v12 = v33;
    v11 = v35;
    v13 = v37;
    v14 = v18;
    goto LABEL_14;
  }
  v20 = sub_16E8CB0();
  v21 = (__m128i *)v20[3];
  v22 = (__int64)v20;
  if ( v20[2] - (_QWORD)v21 <= 0x28u )
  {
    v5 = "WARNING: Internalize couldn't load file '";
    v25 = sub_16E7EE0((__int64)v20, "WARNING: Internalize couldn't load file '", 0x29u);
    v24 = *(__m128i **)(v25 + 24);
    v22 = v25;
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F89B80);
    v21[2].m128i_i8[8] = 39;
    v21[2].m128i_i64[0] = 0x20656C6966206461LL;
    *v21 = si128;
    v21[1] = _mm_load_si128((const __m128i *)&xmmword_3F89B90);
    v24 = (__m128i *)(v20[3] + 41LL);
    v20[3] = v24;
  }
  v26 = *(_QWORD *)(v22 + 16) - (_QWORD)v24;
  if ( v26 < a3 )
  {
    v5 = a2;
    v28 = sub_16E7EE0(v22, a2, a3);
    v24 = *(__m128i **)(v28 + 24);
    v22 = v28;
    v26 = *(_QWORD *)(v28 + 16) - (_QWORD)v24;
  }
  else if ( a3 )
  {
    v5 = a2;
    memcpy(v24, a2, a3);
    v29 = *(_QWORD *)(v22 + 16);
    v24 = (__m128i *)(a3 + *(_QWORD *)(v22 + 24));
    *(_QWORD *)(v22 + 24) = v24;
    v26 = v29 - (_QWORD)v24;
  }
  if ( v26 <= 0x1F )
  {
    v5 = "'! Continuing as if it's empty.\n";
    sub_16E7EE0(v22, "'! Continuing as if it's empty.\n", 0x20u);
  }
  else
  {
    *v24 = _mm_load_si128((const __m128i *)&xmmword_3F89BA0);
    v24[1] = _mm_load_si128((const __m128i *)&xmmword_3F89BB0);
    *(_QWORD *)(v22 + 24) += 32LL;
  }
  v43 = off_4A06540;
  v48[0] = off_4A06568;
  v45[0] = off_4A06448;
  sub_222C7F0(v45);
  sub_2207D90(v47);
  v45[0] = off_4A07480;
  sub_2209150(v46, v5, v27);
  v43 = (__int64 (__fastcall **)())qword_4A064D0;
  *(__int64 (__fastcall ***)())((char *)&v43 + qword_4A064D0[-3]) = (__int64 (__fastcall **)())&unk_4A064F8;
LABEL_16:
  v44 = 0;
  v48[0] = off_4A06798;
  return sub_222E050(v48);
}
