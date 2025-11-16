// Function: sub_206C150
// Address: 0x206c150
//
void __fastcall sub_206C150(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v7; // rax
  int v8; // edx
  __int64 v9; // rax
  int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // r12
  __int64 *v13; // rcx
  __int64 v14; // rbx
  char v15; // dl
  __int64 v16; // r15
  unsigned int v17; // esi
  __int64 v18; // r9
  unsigned int v19; // r8d
  __int64 *v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rdx
  __int64 *v23; // rsi
  __int64 *v24; // rcx
  __int64 **v25; // r13
  __int64 *v26; // rsi
  __int64 *v27; // r12
  __int64 v28; // rdx
  __int64 v29; // r13
  __int64 v30; // rcx
  __int64 v31; // r8
  int v32; // r9d
  __int64 *v33; // rax
  __int64 *v34; // r10
  __int64 v35; // rcx
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  unsigned __int64 v38; // rbx
  bool v39; // zf
  __int64 v40; // rsi
  int v41; // edx
  __int64 *v42; // rbx
  int v43; // r12d
  __int64 *v44; // rcx
  int v45; // eax
  int v46; // eax
  int v47; // r11d
  int v48; // r11d
  __int64 v49; // r10
  unsigned int v50; // edx
  __int64 v51; // r8
  int v52; // edi
  __int64 *v53; // rsi
  int v54; // r11d
  int v55; // r11d
  __int64 v56; // r10
  int v57; // edi
  unsigned int v58; // edx
  __int64 v59; // r8
  __int128 v60; // [rsp-10h] [rbp-1D0h]
  __int64 *v61; // [rsp+8h] [rbp-1B8h]
  __int64 v62; // [rsp+10h] [rbp-1B0h]
  __int64 v63; // [rsp+10h] [rbp-1B0h]
  int v64; // [rsp+24h] [rbp-19Ch]
  unsigned int v65; // [rsp+24h] [rbp-19Ch]
  __int64 v66; // [rsp+28h] [rbp-198h]
  __int64 *v67; // [rsp+28h] [rbp-198h]
  __int64 *v68; // [rsp+28h] [rbp-198h]
  __int64 v69; // [rsp+50h] [rbp-170h] BYREF
  int v70; // [rsp+58h] [rbp-168h]
  __int64 v71; // [rsp+60h] [rbp-160h] BYREF
  __int64 *v72; // [rsp+68h] [rbp-158h]
  __int64 *v73; // [rsp+70h] [rbp-150h]
  __int64 v74; // [rsp+78h] [rbp-148h]
  int v75; // [rsp+80h] [rbp-140h]
  _BYTE v76[312]; // [rsp+88h] [rbp-138h] BYREF

  v7 = *(_QWORD *)(a1 + 712);
  v8 = *(_DWORD *)(a2 + 20);
  v71 = 0;
  v74 = 32;
  v9 = *(_QWORD *)(v7 + 784);
  v10 = v8 & 0xFFFFFFF;
  v75 = 0;
  v62 = v9;
  v11 = (__int64 *)v76;
  v72 = (__int64 *)v76;
  v73 = (__int64 *)v76;
  if ( v10 != 1 )
  {
    v12 = 24;
    v13 = (__int64 *)v76;
    v66 = 24LL * (unsigned int)(v10 - 2) + 48;
    while ( 1 )
    {
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      {
        v14 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + v12);
        if ( v13 != v11 )
          goto LABEL_4;
      }
      else
      {
        v14 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) + v12);
        if ( v13 != v11 )
          goto LABEL_4;
      }
      v23 = &v11[HIDWORD(v74)];
      if ( v23 != v11 )
      {
        v24 = 0;
        do
        {
          if ( v14 == *v11 )
            goto LABEL_9;
          if ( *v11 == -2 )
            v24 = v11;
          ++v11;
        }
        while ( v23 != v11 );
        if ( v24 )
        {
          *v24 = v14;
          --v75;
          ++v71;
          goto LABEL_5;
        }
      }
      if ( HIDWORD(v74) < (unsigned int)v74 )
      {
        ++HIDWORD(v74);
        *v23 = v14;
        ++v71;
        goto LABEL_5;
      }
LABEL_4:
      sub_16CCBA0((__int64)&v71, v14);
      if ( !v15 )
        goto LABEL_9;
LABEL_5:
      v16 = *(_QWORD *)(a1 + 712);
      v17 = *(_DWORD *)(v16 + 72);
      if ( v17 )
      {
        v18 = *(_QWORD *)(v16 + 56);
        v19 = (v17 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v20 = (__int64 *)(v18 + 16LL * v19);
        v21 = *v20;
        if ( v14 == *v20 )
        {
          v22 = v20[1];
          goto LABEL_8;
        }
        v64 = 1;
        v44 = 0;
        while ( v21 != -8 )
        {
          if ( v44 || v21 != -16 )
            v20 = v44;
          v19 = (v17 - 1) & (v64 + v19);
          v61 = (__int64 *)(v18 + 16LL * v19);
          v21 = *v61;
          if ( v14 == *v61 )
          {
            v22 = v61[1];
            goto LABEL_8;
          }
          ++v64;
          v44 = v20;
          v20 = (__int64 *)(v18 + 16LL * v19);
        }
        if ( !v44 )
          v44 = v20;
        v45 = *(_DWORD *)(v16 + 64);
        ++*(_QWORD *)(v16 + 48);
        v46 = v45 + 1;
        if ( 4 * v46 < 3 * v17 )
        {
          if ( v17 - *(_DWORD *)(v16 + 68) - v46 > v17 >> 3 )
            goto LABEL_44;
          v65 = ((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4);
          sub_1D52F30(v16 + 48, v17);
          v54 = *(_DWORD *)(v16 + 72);
          if ( !v54 )
          {
LABEL_74:
            ++*(_DWORD *)(v16 + 64);
            BUG();
          }
          v55 = v54 - 1;
          v56 = *(_QWORD *)(v16 + 56);
          v53 = 0;
          v57 = 1;
          v58 = v55 & v65;
          v46 = *(_DWORD *)(v16 + 64) + 1;
          v44 = (__int64 *)(v56 + 16LL * (v55 & v65));
          v59 = *v44;
          if ( v14 == *v44 )
            goto LABEL_44;
          while ( v59 != -8 )
          {
            if ( v59 == -16 && !v53 )
              v53 = v44;
            v58 = v55 & (v57 + v58);
            v44 = (__int64 *)(v56 + 16LL * v58);
            v59 = *v44;
            if ( v14 == *v44 )
              goto LABEL_44;
            ++v57;
          }
          goto LABEL_60;
        }
      }
      else
      {
        ++*(_QWORD *)(v16 + 48);
      }
      sub_1D52F30(v16 + 48, 2 * v17);
      v47 = *(_DWORD *)(v16 + 72);
      if ( !v47 )
        goto LABEL_74;
      v48 = v47 - 1;
      v49 = *(_QWORD *)(v16 + 56);
      v50 = v48 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v46 = *(_DWORD *)(v16 + 64) + 1;
      v44 = (__int64 *)(v49 + 16LL * v50);
      v51 = *v44;
      if ( v14 == *v44 )
        goto LABEL_44;
      v52 = 1;
      v53 = 0;
      while ( v51 != -8 )
      {
        if ( v51 == -16 && !v53 )
          v53 = v44;
        v50 = v48 & (v52 + v50);
        v44 = (__int64 *)(v49 + 16LL * v50);
        v51 = *v44;
        if ( v14 == *v44 )
          goto LABEL_44;
        ++v52;
      }
LABEL_60:
      if ( v53 )
        v44 = v53;
LABEL_44:
      *(_DWORD *)(v16 + 64) = v46;
      if ( *v44 != -8 )
        --*(_DWORD *)(v16 + 68);
      *v44 = v14;
      v22 = 0;
      v44[1] = 0;
LABEL_8:
      sub_2052F00(a1, v62, v22, -1);
LABEL_9:
      v12 += 24;
      if ( v66 == v12 )
        break;
      v13 = v73;
      v11 = v72;
    }
  }
  sub_1D96570(*(unsigned int **)(v62 + 112), *(unsigned int **)(v62 + 120));
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v25 = *(__int64 ***)(a2 - 8);
  else
    v25 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v26 = *v25;
  v67 = *(__int64 **)(a1 + 552);
  v27 = sub_20685E0(a1, *v25, a3, a4, a5);
  v29 = v28;
  v33 = sub_2051DF0(
          (__int64 *)a1,
          *(double *)a3.m128i_i64,
          *(double *)a4.m128i_i64,
          a5,
          (__int64)v26,
          v28,
          v30,
          v31,
          v32);
  v34 = v67;
  v69 = 0;
  v35 = (__int64)v33;
  v36 = *(_QWORD *)a1;
  v38 = v37;
  v39 = *(_QWORD *)a1 == 0;
  v70 = *(_DWORD *)(a1 + 536);
  if ( !v39 && &v69 != (__int64 *)(v36 + 48) )
  {
    v40 = *(_QWORD *)(v36 + 48);
    v69 = v40;
    if ( v40 )
    {
      v63 = v35;
      sub_1623A60((__int64)&v69, v40, 2);
      v35 = v63;
      v34 = v67;
    }
  }
  *((_QWORD *)&v60 + 1) = v29;
  *(_QWORD *)&v60 = v27;
  v68 = v34;
  v42 = sub_1D332F0(
          v34,
          189,
          (__int64)&v69,
          1,
          0,
          0,
          *(double *)a3.m128i_i64,
          *(double *)a4.m128i_i64,
          a5,
          v35,
          v38,
          v60);
  v43 = v41;
  if ( v42 )
  {
    nullsub_686();
    v68[22] = (__int64)v42;
    *((_DWORD *)v68 + 46) = v43;
    sub_1D23870();
  }
  else
  {
    v68[22] = 0;
    *((_DWORD *)v68 + 46) = v41;
  }
  if ( v69 )
    sub_161E7C0((__int64)&v69, v69);
  if ( v73 != v72 )
    _libc_free((unsigned __int64)v73);
}
