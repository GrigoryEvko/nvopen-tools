// Function: sub_30E3440
// Address: 0x30e3440
//
void __fastcall sub_30E3440(__int64 a1)
{
  void (__fastcall *v2)(__m128i *, __m128i *, __int64); // rax
  unsigned __int64 v3; // rdx
  __int64 v4; // rcx
  unsigned __int64 v5; // rdx
  unsigned int v6; // ecx
  __int64 v7; // rdi
  __int64 v8; // r15
  unsigned int v9; // eax
  __int64 *v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rax
  __int32 v13; // eax
  __int32 v14; // r15d
  bool v15; // zf
  __int32 v16; // edx
  __int32 v17; // eax
  char v18; // bl
  char v19; // al
  void (__fastcall *v20)(__m128i *, __int64, __int64); // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // rsi
  __m128i v23; // xmm1
  __m128i v24; // xmm0
  __int64 v25; // rdi
  unsigned __int64 v26; // rdx
  __int64 *v27; // r15
  __m128i v28; // xmm0
  __m128i v29; // xmm2
  unsigned __int64 v30; // rdx
  __int64 *v31; // rbx
  __int64 v32; // r9
  bool v33; // cc
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // rdi
  char v36; // cl
  int v37; // edx
  unsigned __int64 v38; // r9
  char v39; // cl
  __int32 v40; // r8d
  __int64 v41; // rsi
  unsigned __int64 v42; // rdi
  unsigned __int64 v43; // rdi
  int v44; // esi
  __int64 v45; // rax
  __m128i v46; // xmm0
  __int64 *v47; // r14
  __m128i v48; // xmm3
  unsigned __int64 v49; // rdx
  __int64 *v50; // rbx
  __int64 v51; // r15
  __int32 v52; // [rsp+0h] [rbp-100h]
  __int32 v53; // [rsp+0h] [rbp-100h]
  char v54; // [rsp+7h] [rbp-F9h]
  char v55; // [rsp+7h] [rbp-F9h]
  int v56; // [rsp+8h] [rbp-F8h]
  int v57; // [rsp+8h] [rbp-F8h]
  __int32 v58; // [rsp+8h] [rbp-F8h]
  __int32 v59; // [rsp+Ch] [rbp-F4h]
  __int32 v60; // [rsp+Ch] [rbp-F4h]
  int v61; // [rsp+Ch] [rbp-F4h]
  __int64 v62; // [rsp+10h] [rbp-F0h]
  int v63; // [rsp+10h] [rbp-F0h]
  __int64 v64; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v65; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v66; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v67; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v68; // [rsp+18h] [rbp-E8h]
  __int32 v69; // [rsp+18h] [rbp-E8h]
  __int32 v70; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v71; // [rsp+20h] [rbp-E0h] BYREF
  unsigned int v72; // [rsp+28h] [rbp-D8h]
  __m128i v73; // [rsp+30h] [rbp-D0h] BYREF
  void (__fastcall *v74)(__m128i *, __m128i *, __int64); // [rsp+40h] [rbp-C0h] BYREF
  unsigned __int64 v75; // [rsp+48h] [rbp-B8h]
  char v76; // [rsp+50h] [rbp-B0h]
  __m128i v77; // [rsp+60h] [rbp-A0h] BYREF
  void (__fastcall *v78)(__m128i *, __m128i *, __int64); // [rsp+70h] [rbp-90h]
  unsigned __int64 v79; // [rsp+78h] [rbp-88h] BYREF
  unsigned int v80; // [rsp+80h] [rbp-80h]
  char v81; // [rsp+88h] [rbp-78h]
  __m128i v82; // [rsp+90h] [rbp-70h] BYREF
  void (__fastcall *v83)(__m128i *, __m128i *, __int64); // [rsp+A0h] [rbp-60h]
  unsigned __int64 v84; // [rsp+A8h] [rbp-58h] BYREF
  unsigned __int32 v85; // [rsp+B0h] [rbp-50h]
  unsigned __int64 v86; // [rsp+B8h] [rbp-48h] BYREF
  unsigned int v87; // [rsp+C0h] [rbp-40h]
  char v88; // [rsp+C8h] [rbp-38h]

  v2 = *(void (__fastcall **)(__m128i *, __m128i *, __int64))(a1 + 168);
  v74 = 0;
  if ( !v2 )
  {
    v5 = *(unsigned int *)(a1 + 16);
    v4 = v5;
    if ( v5 <= 1 )
      goto LABEL_6;
LABEL_109:
    v46 = _mm_loadu_si128(&v73);
    v47 = *(__int64 **)(a1 + 8);
    v78 = v2;
    v48 = _mm_loadu_si128(&v82);
    v74 = 0;
    v49 = v75;
    v82 = v46;
    v75 = v84;
    v50 = &v47[v4 - 1];
    v79 = v49;
    v73 = v48;
    v77 = v46;
    v51 = *v50;
    *v50 = *v47;
    v83 = 0;
    if ( v78 )
    {
      v78(&v82, &v77, 2);
      v84 = v79;
      v83 = v78;
    }
    sub_30E32A0((__int64)v47, 0, v50 - v47, v51, (__int64)&v82);
    if ( v83 )
      v83(&v82, &v82, 3);
    if ( v78 )
      v78(&v77, &v77, 3);
LABEL_31:
    v2 = v74;
    goto LABEL_3;
  }
  v2(&v73, (__m128i *)(a1 + 152), 2);
  v3 = *(unsigned int *)(a1 + 16);
  v75 = *(_QWORD *)(a1 + 176);
  v2 = *(void (__fastcall **)(__m128i *, __m128i *, __int64))(a1 + 168);
  v4 = v3;
  v74 = v2;
  if ( v3 > 1 )
    goto LABEL_109;
LABEL_3:
  if ( v2 )
    ((void (__fastcall *)(__m128i *, __m128i *, __int64, __int64))v2)(&v73, &v73, 3, v4 * 8);
  v5 = *(unsigned int *)(a1 + 16);
  while ( 1 )
  {
LABEL_6:
    v6 = *(_DWORD *)(a1 + 240);
    v7 = *(_QWORD *)(a1 + 224);
    v8 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8 * v5 - 8);
    if ( v6 )
    {
      v9 = (v6 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v10 = (__int64 *)(v7 + 56LL * v9);
      v11 = *v10;
      if ( v8 == *v10 )
        goto LABEL_8;
      v44 = 1;
      while ( v11 != -4096 )
      {
        v9 = (v6 - 1) & (v44 + v9);
        v10 = (__int64 *)(v7 + 56LL * v9);
        v11 = *v10;
        if ( v8 == *v10 )
          goto LABEL_8;
        ++v44;
      }
    }
    v10 = (__int64 *)(v7 + 56LL * v6);
LABEL_8:
    v12 = v10[1];
    v81 = 0;
    v77.m128i_i64[0] = v12;
    if ( *((_BYTE *)v10 + 48) )
    {
      LODWORD(v78) = *((_DWORD *)v10 + 6);
      if ( (unsigned int)v78 > 0x40 )
        sub_C43780((__int64)&v77.m128i_i64[1], (const void **)v10 + 2);
      else
        v77.m128i_i64[1] = v10[2];
      v80 = *((_DWORD *)v10 + 10);
      if ( v80 > 0x40 )
        sub_C43780((__int64)&v79, (const void **)v10 + 4);
      else
        v79 = v10[4];
      v81 = 1;
    }
    sub_30E1100((__int64)&v82, v8, *(_QWORD *)(a1 + 248), *(int **)(a1 + 256));
    v76 = 0;
    v13 = v82.m128i_i32[0];
    v14 = v82.m128i_i32[2];
    if ( v88 )
    {
      v73.m128i_i32[2] = v85;
      if ( v85 > 0x40 )
      {
        v70 = v82.m128i_i32[0];
        sub_C43780((__int64)&v73, (const void **)&v84);
        v13 = v70;
      }
      else
      {
        v73.m128i_i64[0] = v84;
      }
      v37 = v87;
      LODWORD(v75) = v87;
      if ( v87 > 0x40 )
      {
        v69 = v13;
        sub_C43780((__int64)&v74, (const void **)&v86);
        v37 = v75;
        v38 = (unsigned __int64)v74;
        v13 = v69;
      }
      else
      {
        v38 = v86;
      }
      v39 = v88;
      v40 = v73.m128i_i32[2];
      v41 = v73.m128i_i64[0];
      if ( v88 )
      {
        v88 = 0;
        if ( v87 > 0x40 && v86 )
        {
          v52 = v73.m128i_i32[2];
          v54 = v39;
          v56 = v37;
          v59 = v13;
          v62 = v73.m128i_i64[0];
          v65 = v38;
          j_j___libc_free_0_0(v86);
          v40 = v52;
          v39 = v54;
          v37 = v56;
          v13 = v59;
          v41 = v62;
          v38 = v65;
        }
        if ( v85 > 0x40 && v84 )
        {
          v53 = v40;
          v55 = v39;
          v57 = v37;
          v60 = v13;
          v66 = v38;
          j_j___libc_free_0_0(v84);
          v40 = v53;
          v39 = v55;
          v37 = v57;
          v13 = v60;
          v38 = v66;
        }
        v15 = *((_BYTE *)v10 + 48) == 0;
        *((_DWORD *)v10 + 2) = v13;
        *((_DWORD *)v10 + 3) = v14;
        if ( v15 )
        {
          if ( !v39 )
          {
LABEL_11:
            v16 = *((_DWORD *)v10 + 2);
            v82.m128i_i32[0] = v16;
            v17 = *((_DWORD *)v10 + 3);
            LOBYTE(v86) = 0;
            v82.m128i_i32[1] = v17;
LABEL_12:
            v18 = v77.m128i_i32[0] + v77.m128i_i32[1] < (int)qword_5030F68;
            v19 = (int)qword_5030F68 > v16 + v17;
            if ( v77.m128i_i32[0] + v77.m128i_i32[1] < (int)qword_5030F68 || v19 )
            {
              if ( v18 != v19 )
                goto LABEL_41;
              v18 = v77.m128i_i32[0] < v16;
            }
            else
            {
              v18 = v81;
              if ( !v81 )
              {
                v18 = v16 > v77.m128i_i32[0];
                goto LABEL_16;
              }
            }
LABEL_34:
            if ( !(_BYTE)v86 )
              goto LABEL_41;
            goto LABEL_35;
          }
LABEL_99:
          v45 = v10[1];
          *((_DWORD *)v10 + 6) = v40;
          v10[2] = v41;
          *((_DWORD *)v10 + 10) = v37;
          v10[4] = v38;
          *((_BYTE *)v10 + 48) = 1;
          v82.m128i_i64[0] = v45;
          LOBYTE(v86) = 0;
          LODWORD(v83) = *((_DWORD *)v10 + 6);
          if ( (unsigned int)v83 <= 0x40 )
            goto LABEL_58;
          goto LABEL_100;
        }
      }
      else
      {
        v15 = *((_BYTE *)v10 + 48) == 0;
        *((_DWORD *)v10 + 2) = v13;
        *((_DWORD *)v10 + 3) = v14;
        if ( v15 )
          goto LABEL_99;
      }
      if ( *((_DWORD *)v10 + 6) > 0x40u )
      {
        v42 = v10[2];
        if ( v42 )
        {
          v58 = v40;
          v61 = v37;
          v67 = v38;
          j_j___libc_free_0_0(v42);
          v40 = v58;
          v37 = v61;
          v38 = v67;
        }
      }
      v33 = *((_DWORD *)v10 + 10) <= 0x40u;
      v10[2] = v41;
      *((_DWORD *)v10 + 6) = v40;
      if ( !v33 )
      {
        v43 = v10[4];
        if ( v43 )
        {
          v63 = v37;
          v68 = v38;
          j_j___libc_free_0_0(v43);
          v37 = v63;
          v38 = v68;
        }
      }
      v10[4] = v38;
      v36 = *((_BYTE *)v10 + 48);
      *((_DWORD *)v10 + 10) = v37;
    }
    else
    {
      v15 = *((_BYTE *)v10 + 48) == 0;
      *((_DWORD *)v10 + 2) = v82.m128i_i32[0];
      *((_DWORD *)v10 + 3) = v14;
      if ( v15 )
        goto LABEL_11;
      v33 = *((_DWORD *)v10 + 10) <= 0x40u;
      *((_BYTE *)v10 + 48) = 0;
      if ( !v33 )
      {
        v34 = v10[4];
        if ( v34 )
          j_j___libc_free_0_0(v34);
      }
      if ( *((_DWORD *)v10 + 6) > 0x40u )
      {
        v35 = v10[2];
        if ( v35 )
          j_j___libc_free_0_0(v35);
      }
      v36 = *((_BYTE *)v10 + 48);
    }
    v16 = *((_DWORD *)v10 + 2);
    v82.m128i_i32[0] = v16;
    v17 = *((_DWORD *)v10 + 3);
    LOBYTE(v86) = 0;
    v82.m128i_i32[1] = v17;
    if ( !v36 )
      goto LABEL_12;
    LODWORD(v83) = *((_DWORD *)v10 + 6);
    if ( (unsigned int)v83 <= 0x40 )
    {
LABEL_58:
      v82.m128i_i64[1] = v10[2];
      v85 = *((_DWORD *)v10 + 10);
      if ( v85 > 0x40 )
        goto LABEL_101;
      goto LABEL_59;
    }
LABEL_100:
    sub_C43780((__int64)&v82.m128i_i64[1], (const void **)v10 + 2);
    v85 = *((_DWORD *)v10 + 10);
    if ( v85 > 0x40 )
    {
LABEL_101:
      sub_C43780((__int64)&v84, (const void **)v10 + 4);
      goto LABEL_60;
    }
LABEL_59:
    v84 = v10[4];
LABEL_60:
    LOBYTE(v86) = 1;
    v18 = v77.m128i_i32[0] + v77.m128i_i32[1] < (int)qword_5030F68;
    if ( v77.m128i_i32[0] + v77.m128i_i32[1] >= (int)qword_5030F68
      && v82.m128i_i32[0] + v82.m128i_i32[1] >= (int)qword_5030F68 )
    {
      v18 = v81;
      if ( v81 )
      {
        sub_C472A0((__int64)&v71, (__int64)&v79, &v82.m128i_i64[1]);
        sub_C472A0((__int64)&v73, (__int64)&v84, &v77.m128i_i64[1]);
        v18 = (int)sub_C49970((__int64)&v71, (unsigned __int64 *)&v73) > 0;
        if ( v73.m128i_i32[2] > 0x40u && v73.m128i_i64[0] )
          j_j___libc_free_0_0(v73.m128i_u64[0]);
        if ( v72 > 0x40 && v71 )
          j_j___libc_free_0_0(v71);
      }
      goto LABEL_34;
    }
    if ( v18 == v82.m128i_i32[0] + v82.m128i_i32[1] < (int)qword_5030F68 )
    {
      v18 = v77.m128i_i32[0] < v82.m128i_i32[0];
      goto LABEL_34;
    }
LABEL_35:
    LOBYTE(v86) = 0;
    if ( v85 > 0x40 && v84 )
      j_j___libc_free_0_0(v84);
    if ( (unsigned int)v83 > 0x40 && v82.m128i_i64[1] )
      j_j___libc_free_0_0(v82.m128i_u64[1]);
LABEL_41:
    if ( v81 )
    {
      v81 = 0;
      if ( v80 > 0x40 && v79 )
        j_j___libc_free_0_0(v79);
      if ( (unsigned int)v78 > 0x40 && v77.m128i_i64[1] )
        break;
    }
LABEL_16:
    if ( !v18 )
      return;
LABEL_17:
    v20 = *(void (__fastcall **)(__m128i *, __int64, __int64))(a1 + 168);
    v21 = v75;
    v74 = 0;
    if ( v20 )
    {
      v20(&v73, a1 + 152, 2);
      v21 = *(_QWORD *)(a1 + 176);
      v20 = *(void (__fastcall **)(__m128i *, __int64, __int64))(a1 + 168);
    }
    v22 = *(unsigned int *)(a1 + 16);
    v23 = _mm_loadu_si128(&v82);
    v78 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v20;
    v24 = _mm_loadu_si128(&v73);
    v79 = v21;
    v25 = *(_QWORD *)(a1 + 8);
    v73 = v23;
    v74 = 0;
    v75 = v84;
    v82 = v24;
    v77 = v24;
    sub_30E31D0(v25, ((8 * v22) >> 3) - 1, 0, *(_QWORD *)(v25 + 8 * v22 - 8), (__int64)&v77);
    if ( v78 )
      v78(&v77, &v77, 3);
    if ( v74 )
      v74(&v73, &v73, 3);
    v2 = *(void (__fastcall **)(__m128i *, __m128i *, __int64))(a1 + 168);
    v74 = 0;
    if ( v2 )
    {
      v2(&v73, (__m128i *)(a1 + 152), 2);
      v26 = *(unsigned int *)(a1 + 16);
      v75 = *(_QWORD *)(a1 + 176);
      v2 = *(void (__fastcall **)(__m128i *, __m128i *, __int64))(a1 + 168);
      v4 = v26;
      v74 = v2;
      if ( v26 <= 1 )
        goto LABEL_3;
LABEL_25:
      v27 = *(__int64 **)(a1 + 8);
      v28 = _mm_loadu_si128(&v73);
      v78 = v2;
      v29 = _mm_loadu_si128(&v82);
      v74 = 0;
      v30 = v75;
      v82 = v28;
      v75 = v84;
      v31 = &v27[v4 - 1];
      v79 = v30;
      v73 = v29;
      v77 = v28;
      v32 = *v31;
      *v31 = *v27;
      v83 = 0;
      if ( v78 )
      {
        v64 = v32;
        v78(&v82, &v77, 2);
        v32 = v64;
        v84 = v79;
        v83 = v78;
      }
      sub_30E32A0((__int64)v27, 0, v31 - v27, v32, (__int64)&v82);
      if ( v83 )
        v83(&v82, &v82, 3);
      if ( v78 )
        v78(&v77, &v77, 3);
      goto LABEL_31;
    }
    v5 = *(unsigned int *)(a1 + 16);
    v4 = v5;
    if ( v5 > 1 )
      goto LABEL_25;
  }
  j_j___libc_free_0_0(v77.m128i_u64[1]);
  if ( v18 )
    goto LABEL_17;
}
