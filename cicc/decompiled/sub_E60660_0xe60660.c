// Function: sub_E60660
// Address: 0xe60660
//
void *__fastcall sub_E60660(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 *v7; // rax
  __int64 v8; // r13
  __int64 v9; // r14
  __int64 v10; // r15
  __int64 *i; // r12
  __int64 v12; // rcx
  unsigned int v13; // eax
  _BYTE *v14; // rsi
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r13
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  _BYTE *v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rcx
  unsigned int v23; // eax
  __int64 v24; // rbx
  __int64 v25; // r14
  __int64 v26; // rax
  size_t v27; // r14
  __int64 v28; // r9
  __int64 v29; // rax
  unsigned __int64 v30; // rcx
  const __m128i *v31; // r14
  unsigned __int64 v32; // rdx
  __int64 v33; // rax
  unsigned __int64 v34; // r8
  __m128i *v35; // rax
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rax
  unsigned __int64 v39; // rcx
  __int64 v40; // rax
  unsigned __int64 v41; // rdx
  __int64 v42; // rcx
  const __m128i *v43; // rdx
  __m128i *v44; // rax
  unsigned int v45; // eax
  int v46; // r13d
  __int64 v47; // r14
  int *v48; // rax
  int v49; // ebx
  unsigned __int64 v50; // r14
  const void *v51; // rsi
  const void *v52; // rsi
  char *v53; // r14
  unsigned int v55; // ebx
  __int64 v56; // [rsp+10h] [rbp-130h]
  __int64 v57; // [rsp+18h] [rbp-128h]
  __int64 v58; // [rsp+20h] [rbp-120h]
  __int64 v59; // [rsp+28h] [rbp-118h]
  __int64 v60; // [rsp+30h] [rbp-110h]
  __int64 v61; // [rsp+38h] [rbp-108h]
  __int16 v62; // [rsp+40h] [rbp-100h]
  __int64 v63; // [rsp+48h] [rbp-F8h]
  unsigned __int8 *src; // [rsp+50h] [rbp-F0h]
  __int16 v65; // [rsp+5Eh] [rbp-E2h]
  int v66; // [rsp+60h] [rbp-E0h]
  __int64 v67; // [rsp+68h] [rbp-D8h]
  unsigned int v68; // [rsp+68h] [rbp-D8h]
  __int64 v69; // [rsp+70h] [rbp-D0h]
  __int64 v70; // [rsp+70h] [rbp-D0h]
  __int64 v71; // [rsp+70h] [rbp-D0h]
  __int64 *v72; // [rsp+78h] [rbp-C8h]
  unsigned int v73; // [rsp+78h] [rbp-C8h]
  int v74; // [rsp+78h] [rbp-C8h]
  __int64 v75; // [rsp+80h] [rbp-C0h] BYREF
  int v76; // [rsp+88h] [rbp-B8h]
  int v77; // [rsp+8Ch] [rbp-B4h]
  __int64 v78; // [rsp+90h] [rbp-B0h]
  _BYTE *v79; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 v80; // [rsp+A8h] [rbp-98h]
  _BYTE v81[32]; // [rsp+B0h] [rbp-90h] BYREF
  _QWORD v82[3]; // [rsp+D0h] [rbp-70h] BYREF
  __int64 v83; // [rsp+E8h] [rbp-58h]
  void *dest; // [rsp+F0h] [rbp-50h]
  __int64 v85; // [rsp+F8h] [rbp-48h]
  __int64 v86; // [rsp+100h] [rbp-40h]

  v3 = a3;
  v5 = *a2;
  *(_QWORD *)(a3 + 48) = 0;
  *(_DWORD *)(a3 + 104) = 0;
  v56 = v5;
  v85 = 0x100000000LL;
  v82[1] = 2;
  v82[2] = 0;
  v82[0] = &unk_49DD288;
  v86 = a3 + 40;
  v83 = 0;
  dest = 0;
  sub_CB5980((__int64)v82, 0, 0, 0);
  v6 = *(unsigned int *)(v3 + 216);
  v79 = v81;
  v80 = 0x400000000LL;
  v7 = *(__int64 **)(v3 + 208);
  v58 = v6;
  v72 = &v7[2 * v6];
  if ( v7 == v72 )
  {
    if ( !v6 )
      goto LABEL_47;
    v20 = v81;
    goto LABEL_11;
  }
  v8 = 0;
  v67 = v3;
  v9 = *v7;
  v10 = v7[1];
  for ( i = v7 + 2; ; i += 2 )
  {
    v14 = (_BYTE *)v9;
    v17 = (sub_E5F570(a2, v9, v10) << 32) | v8;
    v18 = (unsigned int)v80;
    v19 = (unsigned int)v80 + 1LL;
    if ( v19 > HIDWORD(v80) )
    {
      v14 = v81;
      sub_C8D5F0((__int64)&v79, v81, v19, 8u, v15, v16);
      v18 = (unsigned int)v80;
    }
    *(_QWORD *)&v79[8 * v18] = v17;
    LODWORD(v80) = v80 + 1;
    if ( v72 == i )
      break;
    v9 = *i;
    v12 = i[1];
    v8 = 0;
    if ( v10 )
    {
      v69 = i[1];
      v13 = sub_E5F570(a2, v10, *i);
      v12 = v69;
      v8 = v13;
    }
    v10 = v12;
  }
  v3 = v67;
  v20 = v79;
  v58 = *(unsigned int *)(v67 + 216);
  if ( *(_DWORD *)(v67 + 216) )
  {
    v7 = *(__int64 **)(v67 + 208);
LABEL_11:
    v21 = 0;
    v59 = v3 + 96;
    while ( 1 )
    {
      v22 = v21 + 1;
      v60 = v21 + 1;
      v61 = v7[2 * v21];
      v57 = 8 * v21;
      v23 = *(_DWORD *)&v20[8 * v21 + 4];
      if ( v21 + 1 == v58 )
      {
        v55 = *(_DWORD *)&v20[8 * v21 + 4];
        v63 = v58;
      }
      else
      {
        do
        {
          v55 = v23;
          v23 += *(_DWORD *)&v20[8 * v22] + *(_DWORD *)&v20[8 * v22 + 4];
          if ( v23 > 0xF000 )
          {
            v63 = v22;
            goto LABEL_17;
          }
          ++v22;
        }
        while ( v22 != v58 );
        v63 = v22;
        v55 = v23;
      }
LABEL_17:
      v73 = v55;
      v24 = v3;
      v68 = 0;
      v62 = 4 * (v63 + ~(_WORD)v21);
      do
      {
        if ( v73 > 0xEFFF )
        {
          v66 = 61440;
          v73 -= 61440;
          v65 = -4096;
        }
        else
        {
          v45 = v73;
          v73 = 0;
          v65 = v45;
          v66 = v45;
        }
        v25 = sub_E808D0(v61, 0, v56, 0);
        v26 = sub_E81A90(v68, v56, 0, 0);
        v70 = sub_E81A00(0, v25, v26, v56, 0);
        v27 = *(_QWORD *)(v24 + 264);
        src = *(unsigned __int8 **)(v24 + 256);
        LOWORD(v75) = v62 + v27 + 8;
        sub_CB6200((__int64)v82, (unsigned __int8 *)&v75, 2u);
        if ( v27 > v83 - (__int64)dest )
        {
          sub_CB6200((__int64)v82, src, v27);
        }
        else if ( v27 )
        {
          memcpy(dest, src, v27);
          dest = (char *)dest + v27;
        }
        v29 = *(_QWORD *)(v24 + 48);
        v30 = *(unsigned int *)(v24 + 108);
        v31 = (const __m128i *)&v75;
        v77 = 12;
        v32 = *(_QWORD *)(v24 + 96);
        v78 = 0;
        v76 = v29;
        v33 = *(unsigned int *)(v24 + 104);
        v75 = v70;
        v34 = v33 + 1;
        if ( v33 + 1 > v30 )
        {
          v52 = (const void *)(v24 + 112);
          if ( v32 > (unsigned __int64)&v75 || (unsigned __int64)&v75 >= v32 + 24 * v33 )
          {
            sub_C8D5F0(v59, v52, v34, 0x18u, v34, v28);
            v32 = *(_QWORD *)(v24 + 96);
            v33 = *(unsigned int *)(v24 + 104);
          }
          else
          {
            v53 = (char *)&v75 - v32;
            sub_C8D5F0(v59, v52, v34, 0x18u, v34, v28);
            v32 = *(_QWORD *)(v24 + 96);
            v33 = *(unsigned int *)(v24 + 104);
            v31 = (const __m128i *)&v53[v32];
          }
        }
        v35 = (__m128i *)(v32 + 24 * v33);
        *v35 = _mm_loadu_si128(v31);
        v35[1].m128i_i64[0] = v31[1].m128i_i64[0];
        ++*(_DWORD *)(v24 + 104);
        LODWORD(v75) = 0;
        sub_CB6200((__int64)v82, (unsigned __int8 *)&v75, 4u);
        v38 = *(_QWORD *)(v24 + 48);
        v77 = 11;
        v78 = 0;
        v39 = *(unsigned int *)(v24 + 108);
        v76 = v38;
        v40 = *(unsigned int *)(v24 + 104);
        v75 = v70;
        v41 = v40 + 1;
        if ( v40 + 1 > v39 )
        {
          v50 = *(_QWORD *)(v24 + 96);
          v51 = (const void *)(v24 + 112);
          if ( v50 > (unsigned __int64)&v75 || (unsigned __int64)&v75 >= v50 + 24 * v40 )
          {
            sub_C8D5F0(v59, v51, v41, 0x18u, v36, v37);
            v42 = *(_QWORD *)(v24 + 96);
            v40 = *(unsigned int *)(v24 + 104);
            v43 = (const __m128i *)&v75;
          }
          else
          {
            sub_C8D5F0(v59, v51, v41, 0x18u, v36, v37);
            v42 = *(_QWORD *)(v24 + 96);
            v40 = *(unsigned int *)(v24 + 104);
            v43 = (const __m128i *)((char *)&v75 + v42 - v50);
          }
        }
        else
        {
          v42 = *(_QWORD *)(v24 + 96);
          v43 = (const __m128i *)&v75;
        }
        v44 = (__m128i *)(v42 + 24 * v40);
        *v44 = _mm_loadu_si128(v43);
        v44[1].m128i_i64[0] = v43[1].m128i_i64[0];
        ++*(_DWORD *)(v24 + 104);
        LOWORD(v75) = 0;
        sub_CB6200((__int64)v82, (unsigned __int8 *)&v75, 2u);
        v14 = &v75;
        LOWORD(v75) = v65;
        sub_CB6200((__int64)v82, (unsigned __int8 *)&v75, 2u);
        v68 += v66;
      }
      while ( v73 );
      v20 = v79;
      v3 = v24;
      v46 = *(_DWORD *)&v79[v57 + 4];
      if ( v63 != v60 )
      {
        v71 = v24;
        v47 = v60;
        while ( 1 )
        {
          v48 = (int *)&v20[8 * v47];
          v49 = *v48;
          LODWORD(v48) = v48[1];
          LOWORD(v75) = v46;
          ++v47;
          v74 = (int)v48;
          sub_CB6200((__int64)v82, (unsigned __int8 *)&v75, 2u);
          v14 = &v75;
          LOWORD(v75) = v49;
          sub_CB6200((__int64)v82, (unsigned __int8 *)&v75, 2u);
          v46 += v74 + v49;
          if ( v63 == v47 )
            break;
          v20 = v79;
        }
        v60 = v47;
        v20 = v79;
        v3 = v71;
      }
      v21 = v60;
      if ( v58 == v60 )
        break;
      v7 = *(__int64 **)(v3 + 208);
    }
  }
  if ( v20 != v81 )
    _libc_free(v20, v14);
LABEL_47:
  v82[0] = &unk_49DD388;
  return sub_CB5840((__int64)v82);
}
