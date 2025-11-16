// Function: sub_2CE7030
// Address: 0x2ce7030
//
void __fastcall sub_2CE7030(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rsi
  __int64 v6; // rdx
  int v7; // r8d
  __int64 v8; // r15
  unsigned int v9; // r12d
  __int64 *v10; // rax
  __int64 v11; // rcx
  __int64 *v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 *v15; // r9
  __int64 *v16; // r8
  __int64 v17; // rax
  __int64 v18; // rax
  char *v19; // rcx
  size_t v20; // r13
  __m128i v21; // xmm0
  __int64 *v22; // rax
  __int64 v23; // r12
  __int64 v24; // rsi
  __int64 v25; // rsi
  unsigned __int8 *v26; // rsi
  __int64 v27; // r9
  unsigned int v28; // r13d
  int i; // r11d
  int v30; // r10d
  __int64 *v31; // r11
  __int64 *v32; // r10
  int v33; // r11d
  __int64 *v34; // r9
  __int64 v35; // rdi
  int v36; // eax
  int v37; // edx
  int v38; // eax
  int v39; // eax
  __int64 v40; // rsi
  unsigned int v41; // r12d
  __int64 v42; // rcx
  int v43; // r8d
  __int64 *v44; // rdi
  int v45; // eax
  int v46; // eax
  __int64 v47; // rsi
  int v48; // r8d
  unsigned int v49; // r12d
  __int64 v50; // rcx
  char *v51; // [rsp+0h] [rbp-1D0h]
  void *src; // [rsp+8h] [rbp-1C8h]
  __int64 *v53; // [rsp+18h] [rbp-1B8h]
  __int64 *v54; // [rsp+30h] [rbp-1A0h]
  __int64 v55; // [rsp+48h] [rbp-188h] BYREF
  __m128i *v56; // [rsp+50h] [rbp-180h]
  __int64 v57; // [rsp+58h] [rbp-178h]
  __m128i v58; // [rsp+60h] [rbp-170h] BYREF
  _QWORD *v59; // [rsp+70h] [rbp-160h]
  __int64 v60; // [rsp+78h] [rbp-158h]
  _QWORD v61[4]; // [rsp+80h] [rbp-150h] BYREF
  char v62[32]; // [rsp+A0h] [rbp-130h] BYREF
  __int16 v63; // [rsp+C0h] [rbp-110h]
  __int64 v64[2]; // [rsp+D0h] [rbp-100h] BYREF
  __m128i v65; // [rsp+E0h] [rbp-F0h] BYREF
  char *v66; // [rsp+F0h] [rbp-E0h]
  char *v67; // [rsp+F8h] [rbp-D8h]
  char *v68; // [rsp+100h] [rbp-D0h]
  unsigned int *v69[2]; // [rsp+110h] [rbp-C0h] BYREF
  char v70; // [rsp+120h] [rbp-B0h] BYREF
  void *v71; // [rsp+190h] [rbp-40h]

  v5 = *(unsigned int *)(a1 + 248);
  v6 = *(_QWORD *)(a1 + 232);
  if ( !(_DWORD)v5 )
    return;
  v7 = v5 - 1;
  v8 = a1;
  v9 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  LODWORD(a1) = (v5 - 1) & v9;
  v10 = (__int64 *)(v6 + 48LL * (unsigned int)a1);
  v11 = *v10;
  if ( *v10 != a2 )
  {
    v27 = *v10;
    v28 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    for ( i = 1; ; i = v30 )
    {
      if ( v27 == -4096 )
        return;
      v30 = i + 1;
      v28 = v7 & (i + v28);
      v31 = (__int64 *)(v6 + 48LL * v28);
      v27 = *v31;
      if ( *v31 == a2 )
        break;
    }
    if ( v31 == (__int64 *)(v6 + 48LL * (unsigned int)v5) )
      return;
    v32 = (__int64 *)(v6 + 48LL * (v7 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))));
    v33 = 1;
    v34 = 0;
    while ( v11 != -4096 )
    {
      if ( v34 || v11 != -8192 )
        v32 = v34;
      a1 = v7 & (unsigned int)(a1 + v33);
      v10 = (__int64 *)(v6 + 48 * a1);
      v11 = *v10;
      if ( *v10 == a2 )
        goto LABEL_4;
      ++v33;
      v34 = v32;
      v32 = (__int64 *)(v6 + 48 * a1);
    }
    v35 = v8 + 224;
    if ( !v34 )
      v34 = v10;
    v36 = *(_DWORD *)(v8 + 240);
    ++*(_QWORD *)(v8 + 224);
    v37 = v36 + 1;
    if ( 4 * (v36 + 1) >= (unsigned int)(3 * v5) )
    {
      sub_2CE6D00(v35, 2 * v5);
      v38 = *(_DWORD *)(v8 + 248);
      if ( v38 )
      {
        v39 = v38 - 1;
        v40 = *(_QWORD *)(v8 + 232);
        v41 = v39 & v9;
        v34 = (__int64 *)(v40 + 48LL * v41);
        v37 = *(_DWORD *)(v8 + 240) + 1;
        v42 = *v34;
        if ( *v34 == a2 )
          goto LABEL_41;
        v43 = 1;
        v44 = 0;
        while ( v42 != -4096 )
        {
          if ( v42 == -8192 && !v44 )
            v44 = v34;
          v41 = v39 & (v43 + v41);
          v34 = (__int64 *)(v40 + 48LL * v41);
          v42 = *v34;
          if ( *v34 == a2 )
            goto LABEL_41;
          ++v43;
        }
LABEL_51:
        if ( v44 )
          v34 = v44;
        goto LABEL_41;
      }
    }
    else
    {
      if ( (int)v5 - *(_DWORD *)(v8 + 244) - v37 > (unsigned int)v5 >> 3 )
      {
LABEL_41:
        *(_DWORD *)(v8 + 240) = v37;
        if ( *v34 != -4096 )
          --*(_DWORD *)(v8 + 244);
        *v34 = a2;
        v34[1] = (__int64)(v34 + 3);
        v34[2] = 0x100000000LL;
        return;
      }
      sub_2CE6D00(v35, v5);
      v45 = *(_DWORD *)(v8 + 248);
      if ( v45 )
      {
        v46 = v45 - 1;
        v47 = *(_QWORD *)(v8 + 232);
        v48 = 1;
        v44 = 0;
        v49 = v46 & v9;
        v34 = (__int64 *)(v47 + 48LL * v49);
        v37 = *(_DWORD *)(v8 + 240) + 1;
        v50 = *v34;
        if ( *v34 == a2 )
          goto LABEL_41;
        while ( v50 != -4096 )
        {
          if ( v50 == -8192 && !v44 )
            v44 = v34;
          v49 = v46 & (v48 + v49);
          v34 = (__int64 *)(v47 + 48LL * v49);
          v50 = *v34;
          if ( *v34 == a2 )
            goto LABEL_41;
          ++v48;
        }
        goto LABEL_51;
      }
    }
    ++*(_DWORD *)(v8 + 240);
    BUG();
  }
  if ( v10 != (__int64 *)(48 * v5 + v6) )
  {
LABEL_4:
    v12 = (__int64 *)v10[1];
    v54 = &v12[3 * *((unsigned int *)v10 + 4)];
    if ( v12 != v54 )
    {
      while ( 1 )
      {
        v17 = v12[1];
        v61[0] = a3;
        v61[1] = v17;
        v59 = v61;
        v60 = 0x300000002LL;
        if ( v12[2] )
        {
          v61[2] = v12[2];
          LODWORD(v60) = 3;
        }
        if ( !(unsigned __int8)sub_B19DB0(*(_QWORD *)(v8 + 360), a3, *v12) )
          goto LABEL_11;
        v18 = sub_B46B10(*v12, 0);
        sub_23D0AB0((__int64)v69, v18, 0, 0, 0);
        v19 = 0;
        v63 = 257;
        strcpy(v58.m128i_i8, "align");
        v20 = 8LL * (unsigned int)v60;
        v21 = _mm_load_si128(&v58);
        v56 = &v58;
        src = v59;
        v64[0] = (__int64)&v65;
        v64[1] = 5;
        v57 = 0;
        v58.m128i_i8[0] = 0;
        v66 = 0;
        v67 = 0;
        v68 = 0;
        v65 = v21;
        if ( v20 )
        {
          v66 = (char *)sub_22077B0(8LL * (unsigned int)v60);
          v68 = &v66[v20];
          v51 = &v66[v20];
          memcpy(v66, src, v20);
          v19 = v51;
        }
        v67 = v19;
        v22 = (__int64 *)sub_BD5C60(*v12);
        v55 = sub_ACD6D0(v22);
        v23 = sub_B33530(
                v69,
                *(_QWORD *)(*v12 + 80),
                *(_QWORD *)(*v12 - 32),
                (int)&v55,
                1,
                (__int64)v62,
                (__int64)v64,
                1,
                0);
        if ( v66 )
          j_j___libc_free_0((unsigned __int64)v66);
        if ( (__m128i *)v64[0] != &v65 )
          j_j___libc_free_0(v64[0]);
        if ( v56 != &v58 )
          j_j___libc_free_0((unsigned __int64)v56);
        v16 = (__int64 *)(v23 + 48);
        v24 = *(_QWORD *)(*v12 + 48);
        v64[0] = v24;
        if ( v24 )
          break;
        if ( v16 != v64 )
        {
          v25 = *(_QWORD *)(v23 + 48);
          if ( v25 )
            goto LABEL_28;
        }
LABEL_9:
        *(_QWORD *)(v23 + 72) = *(_QWORD *)(*v12 + 72);
        sub_CFEAE0(*(_QWORD *)(v8 + 368), v23, v13, v14, (__int64)v16, v15);
        nullsub_61();
        v71 = &unk_49DA100;
        nullsub_63();
        if ( (char *)v69[0] != &v70 )
          _libc_free((unsigned __int64)v69[0]);
LABEL_11:
        if ( v59 != v61 )
          _libc_free((unsigned __int64)v59);
        v12 += 3;
        if ( v54 == v12 )
          return;
      }
      sub_B96E90((__int64)v64, v24, 1);
      v16 = (__int64 *)(v23 + 48);
      if ( (__int64 *)(v23 + 48) == v64 )
      {
        if ( v64[0] )
          sub_B91220((__int64)v64, v64[0]);
        goto LABEL_9;
      }
      v25 = *(_QWORD *)(v23 + 48);
      if ( v25 )
      {
LABEL_28:
        v53 = v16;
        sub_B91220((__int64)v16, v25);
        v16 = v53;
      }
      v26 = (unsigned __int8 *)v64[0];
      *(_QWORD *)(v23 + 48) = v64[0];
      if ( v26 )
        sub_B976B0((__int64)v64, v26, (__int64)v16);
      goto LABEL_9;
    }
  }
}
