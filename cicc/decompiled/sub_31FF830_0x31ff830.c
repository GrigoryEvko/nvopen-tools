// Function: sub_31FF830
// Address: 0x31ff830
//
__int64 __fastcall sub_31FF830(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 *v4; // rax
  unsigned int v5; // ecx
  size_t v6; // r14
  size_t v7; // rdx
  size_t v8; // r15
  unsigned int v9; // r12d
  __int64 v10; // rdi
  int v11; // esi
  __int64 v12; // r10
  int v13; // edx
  int v14; // eax
  __int64 v15; // r9
  size_t v16; // rdx
  int v17; // r11d
  unsigned int i; // r8d
  const void *v19; // rcx
  int v20; // eax
  int v22; // eax
  void *v23; // r8
  size_t v24; // r9
  __int64 v25; // rdx
  unsigned int v26; // r8d
  __int64 v27; // r10
  __int64 v28; // rsi
  size_t v29; // r9
  _QWORD *v30; // rdi
  char *v31; // r8
  void *v32; // rax
  __int64 v33; // rax
  size_t v34; // rdx
  size_t v35; // r9
  size_t v36; // rdi
  __int64 v37; // rcx
  int v38; // eax
  unsigned __int64 v39; // r9
  __int64 j; // r8
  int v41; // eax
  int v42; // edx
  __int64 v43; // rax
  __int64 v44; // [rsp+0h] [rbp-B0h]
  size_t v45; // [rsp+8h] [rbp-A8h]
  const void *v46; // [rsp+10h] [rbp-A0h]
  int v47; // [rsp+20h] [rbp-90h]
  unsigned __int8 *v48; // [rsp+20h] [rbp-90h]
  __int64 v49; // [rsp+28h] [rbp-88h]
  size_t v50; // [rsp+28h] [rbp-88h]
  size_t v51; // [rsp+28h] [rbp-88h]
  __int64 v52; // [rsp+28h] [rbp-88h]
  unsigned int v53; // [rsp+30h] [rbp-80h]
  __int64 v54; // [rsp+30h] [rbp-80h]
  unsigned int v55; // [rsp+30h] [rbp-80h]
  size_t v56; // [rsp+30h] [rbp-80h]
  __int64 v57; // [rsp+30h] [rbp-80h]
  unsigned int v58; // [rsp+40h] [rbp-70h]
  void *v59; // [rsp+40h] [rbp-70h]
  void *v60; // [rsp+40h] [rbp-70h]
  __int64 v61; // [rsp+48h] [rbp-68h]
  __int64 v62; // [rsp+48h] [rbp-68h]
  __int64 v63; // [rsp+48h] [rbp-68h]
  __int64 v64; // [rsp+48h] [rbp-68h]
  __int64 v65; // [rsp+58h] [rbp-58h] BYREF
  size_t n[2]; // [rsp+60h] [rbp-50h] BYREF
  _DWORD v67[16]; // [rsp+70h] [rbp-40h] BYREF

  v4 = sub_31FD260((_QWORD *)a1, a2);
  v5 = *(_DWORD *)(a1 + 1128);
  v6 = (size_t)v4;
  LODWORD(v4) = *(_DWORD *)(a1 + 1120);
  n[1] = v7;
  v8 = v7;
  n[0] = v6;
  v9 = (_DWORD)v4 + 1;
  v67[0] = (_DWORD)v4 + 1;
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 1104);
    v10 = a1 + 1104;
    v65 = 0;
    goto LABEL_3;
  }
  v53 = v5;
  v61 = *(_QWORD *)(a1 + 1112);
  v14 = sub_C94890((_QWORD *)n[0], n[1]);
  v15 = 0;
  v16 = n[1];
  v17 = 1;
  for ( i = (v53 - 1) & v14; ; i = (v53 - 1) & v26 )
  {
    v12 = v61 + 24LL * i;
    v19 = *(const void **)v12;
    if ( *(_QWORD *)v12 == -1 )
      break;
    if ( v19 == (const void *)-2LL )
    {
      if ( n[0] == -2 )
        return *(unsigned int *)(v12 + 16);
    }
    else
    {
      if ( *(_QWORD *)(v12 + 8) != v16 )
        goto LABEL_25;
      v47 = v17;
      v49 = v15;
      v58 = i;
      if ( !v16 )
        return *(unsigned int *)(v12 + 16);
      v44 = v61 + 24LL * i;
      v45 = v16;
      v46 = *(const void **)v12;
      v20 = memcmp((const void *)n[0], v19, v16);
      i = v58;
      v19 = v46;
      v16 = v45;
      v12 = v44;
      v15 = v49;
      v17 = v47;
      if ( !v20 )
        return *(unsigned int *)(v12 + 16);
    }
    if ( v19 == (const void *)-2LL && !v15 )
      v15 = v12;
LABEL_25:
    v26 = v17 + i;
    ++v17;
  }
  if ( n[0] != -1 )
  {
    v22 = *(_DWORD *)(a1 + 1120);
    v5 = *(_DWORD *)(a1 + 1128);
    v10 = a1 + 1104;
    if ( v15 )
      v12 = v15;
    ++*(_QWORD *)(a1 + 1104);
    v13 = v22 + 1;
    v65 = v12;
    if ( 4 * (v22 + 1) < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(a1 + 1124) - v13 > v5 >> 3 )
      {
LABEL_17:
        *(_DWORD *)(a1 + 1120) = v13;
        if ( *(_QWORD *)v12 != -1 )
          --*(_DWORD *)(a1 + 1124);
        v23 = 0;
        v24 = 0;
        v25 = 0;
        *(__m128i *)v12 = _mm_loadu_si128((const __m128i *)n);
        *(_DWORD *)(v12 + 16) = v67[0];
        if ( !*(_BYTE *)(a2 + 32) )
        {
LABEL_20:
          v62 = v12;
          (*(void (__fastcall **)(_QWORD, _QWORD, size_t, size_t, void *, size_t, __int64))(**(_QWORD **)(a1 + 528)
                                                                                          + 712LL))(
            *(_QWORD *)(a1 + 528),
            v9,
            v6,
            v8,
            v23,
            v24,
            v25);
          v12 = v62;
          return *(unsigned int *)(v12 + 16);
        }
        v63 = v12;
        sub_B91420(*(_QWORD *)(a2 + 24));
        v27 = v63;
        if ( !*(_BYTE *)(a2 + 32) )
        {
          LOBYTE(v67[0]) = 0;
          v28 = 0;
          v29 = 0;
          n[0] = (size_t)v67;
          n[1] = 0;
          goto LABEL_28;
        }
        v33 = sub_B91420(*(_QWORD *)(a2 + 24));
        LOBYTE(v67[0]) = 0;
        v27 = v63;
        v48 = (unsigned __int8 *)v33;
        v29 = v34;
        v52 = v33;
        n[0] = (size_t)v67;
        n[1] = 0;
        if ( v34 )
        {
          v56 = v34;
          sub_22410F0(n, (v34 + 1) >> 1, 0);
          v35 = v56;
          v36 = n[0];
          v27 = v63;
          v37 = v52;
          if ( (v56 & 1) != 0 )
          {
            v38 = (__int16)word_3F64060[*v48];
            if ( v38 != -1 )
            {
              *(_BYTE *)n[0] = v38;
              v35 = v56 - 1;
              v37 = v52 + 1;
              ++v36;
              goto LABEL_40;
            }
          }
          else
          {
LABEL_40:
            v39 = v35 >> 1;
            if ( v39 )
            {
              for ( j = 0; j != v39; ++j )
              {
                v41 = (__int16)word_3F64060[*(unsigned __int8 *)(v37 + 2 * j)];
                v42 = (__int16)word_3F64060[*(unsigned __int8 *)(v37 + 2 * j + 1)];
                if ( v41 == -1 )
                  break;
                if ( v42 == -1 )
                  break;
                *(_BYTE *)(v36 + j) = v42 | (16 * v41);
              }
            }
          }
          v29 = n[1];
          v28 = LODWORD(n[1]);
        }
        else
        {
          v28 = 0;
        }
LABEL_28:
        v30 = *(_QWORD **)(*(_QWORD *)(a1 + 528) + 8LL);
        v31 = (char *)v30[24];
        v30[34] += v28;
        if ( v30[25] >= (unsigned __int64)&v31[v28] && v31 )
        {
          v30[24] = &v31[v28];
        }
        else
        {
          v57 = v27;
          v43 = sub_9D1E70((__int64)(v30 + 24), v28, v28, 0);
          v29 = n[1];
          v27 = v57;
          v31 = (char *)v43;
        }
        v54 = v27;
        v32 = memcpy(v31, (const void *)n[0], v29);
        v25 = 0;
        v24 = n[1];
        v12 = v54;
        v23 = v32;
        if ( *(_BYTE *)(a2 + 32) )
        {
          v50 = n[1];
          v59 = v32;
          sub_B91420(*(_QWORD *)(a2 + 24));
          v25 = *(unsigned int *)(a2 + 16);
          v24 = v50;
          v23 = v59;
          v12 = v54;
          if ( (unsigned int)(v25 - 1) >= 3 )
            v25 = 0;
        }
        if ( (_DWORD *)n[0] != v67 )
        {
          v51 = v24;
          v60 = v23;
          v55 = v25;
          v64 = v12;
          j_j___libc_free_0(n[0]);
          v24 = v51;
          v23 = v60;
          v25 = v55;
          v12 = v64;
        }
        goto LABEL_20;
      }
      v11 = v5;
LABEL_4:
      sub_1253750(v10, v11);
      sub_262D160(v10, (__int64)n, &v65);
      v12 = v65;
      v13 = *(_DWORD *)(a1 + 1120) + 1;
      goto LABEL_17;
    }
LABEL_3:
    v11 = 2 * v5;
    goto LABEL_4;
  }
  return *(unsigned int *)(v12 + 16);
}
