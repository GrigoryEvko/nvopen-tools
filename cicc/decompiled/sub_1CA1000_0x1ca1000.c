// Function: sub_1CA1000
// Address: 0x1ca1000
//
void __fastcall sub_1CA1000(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rsi
  int v5; // r9d
  __int64 v6; // rdx
  int v8; // ecx
  unsigned int v9; // r12d
  __int64 v10; // r8
  unsigned int v11; // r13d
  __int64 *v12; // rax
  __int64 v13; // rdi
  __int64 v14; // r11
  __int64 *v15; // rdx
  __int64 *i; // r13
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // rax
  unsigned __int8 *v20; // rsi
  __int64 v21; // rax
  size_t v22; // r14
  __m128i v23; // xmm0
  char *v24; // rax
  char *v25; // r12
  __int64 *v26; // rax
  void *v27; // r11
  __int64 v28; // r14
  _QWORD *v29; // rax
  __int64 v30; // r12
  char v31; // al
  int v32; // r14d
  __int64 *v33; // r14
  __int64 v34; // rax
  __int64 v35; // rcx
  unsigned __int8 **v36; // r14
  __int64 v37; // rsi
  unsigned __int8 *v38; // rsi
  unsigned __int8 *v39; // rsi
  __int64 v40; // rsi
  unsigned __int8 *v41; // rsi
  __int64 *v42; // r14
  int v43; // r10d
  __int64 *v44; // rcx
  int v45; // eax
  int v46; // edx
  int v47; // r8d
  int v48; // r8d
  __int64 v49; // r9
  unsigned int v50; // r12d
  __int64 v51; // rdi
  int v52; // esi
  __int64 *v53; // rax
  int v54; // edi
  int v55; // edi
  __int64 v56; // r9
  int v57; // esi
  unsigned int v58; // r12d
  __int64 v59; // r8
  __int64 v60; // [rsp+0h] [rbp-1B0h]
  void *srcb; // [rsp+20h] [rbp-190h]
  void *src; // [rsp+20h] [rbp-190h]
  void *srca; // [rsp+20h] [rbp-190h]
  __int64 *v64; // [rsp+28h] [rbp-188h]
  __int64 v66; // [rsp+58h] [rbp-158h] BYREF
  __int64 v67; // [rsp+60h] [rbp-150h] BYREF
  __int16 v68; // [rsp+70h] [rbp-140h]
  _QWORD v69[2]; // [rsp+80h] [rbp-130h] BYREF
  __int16 v70; // [rsp+90h] [rbp-120h]
  __m128i *v71; // [rsp+A0h] [rbp-110h]
  __int64 v72; // [rsp+A8h] [rbp-108h]
  __m128i v73; // [rsp+B0h] [rbp-100h] BYREF
  _QWORD *v74; // [rsp+C0h] [rbp-F0h]
  __int64 v75; // [rsp+C8h] [rbp-E8h]
  _QWORD v76[4]; // [rsp+D0h] [rbp-E0h] BYREF
  unsigned __int8 *v77[2]; // [rsp+F0h] [rbp-C0h] BYREF
  __m128i v78; // [rsp+100h] [rbp-B0h] BYREF
  char *v79; // [rsp+110h] [rbp-A0h]
  char *v80; // [rsp+118h] [rbp-98h]
  char *v81; // [rsp+120h] [rbp-90h]
  unsigned __int8 *v82; // [rsp+130h] [rbp-80h] BYREF
  __int64 v83; // [rsp+138h] [rbp-78h]
  __int64 *v84; // [rsp+140h] [rbp-70h]
  __int64 v85; // [rsp+148h] [rbp-68h]
  __int64 v86; // [rsp+150h] [rbp-60h]
  int v87; // [rsp+158h] [rbp-58h]
  __int64 v88; // [rsp+160h] [rbp-50h]
  __int64 v89; // [rsp+168h] [rbp-48h]

  v4 = *(unsigned int *)(a1 + 248);
  if ( !(_DWORD)v4 )
    return;
  v5 = v4 - 1;
  v6 = *(_QWORD *)(a1 + 232);
  v8 = 1;
  v9 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  LODWORD(v10) = (v4 - 1) & v9;
  v11 = v10;
  v12 = (__int64 *)(v6 + 48LL * (unsigned int)v10);
  v13 = *v12;
  v14 = *v12;
  if ( *v12 != a2 )
  {
    while ( 1 )
    {
      if ( v14 == -8 )
        return;
      v11 = v5 & (v8 + v11);
      v42 = (__int64 *)(v6 + 48LL * v11);
      v14 = *v42;
      if ( *v42 == a2 )
        break;
      ++v8;
    }
    v43 = 1;
    v44 = 0;
    if ( v42 == (__int64 *)(v6 + 48LL * (unsigned int)v4) )
      return;
    while ( v13 != -8 )
    {
      if ( !v44 && v13 == -16 )
        v44 = v12;
      v10 = v5 & (unsigned int)(v10 + v43);
      v12 = (__int64 *)(v6 + 48 * v10);
      v13 = *v12;
      if ( *v12 == a2 )
        goto LABEL_4;
      ++v43;
    }
    if ( !v44 )
      v44 = v12;
    v45 = *(_DWORD *)(a1 + 240);
    ++*(_QWORD *)(a1 + 224);
    v46 = v45 + 1;
    if ( 4 * (v45 + 1) >= (unsigned int)(3 * v4) )
    {
      sub_1CA0D00(a1 + 224, 2 * v4);
      v47 = *(_DWORD *)(a1 + 248);
      if ( v47 )
      {
        v48 = v47 - 1;
        v49 = *(_QWORD *)(a1 + 232);
        v50 = v48 & v9;
        v44 = (__int64 *)(v49 + 48LL * v50);
        v46 = *(_DWORD *)(a1 + 240) + 1;
        v51 = *v44;
        if ( *v44 == a2 )
          goto LABEL_66;
        v52 = 1;
        v53 = 0;
        while ( v51 != -8 )
        {
          if ( v51 == -16 && !v53 )
            v53 = v44;
          v50 = v48 & (v52 + v50);
          v44 = (__int64 *)(v49 + 48LL * v50);
          v51 = *v44;
          if ( *v44 == a2 )
            goto LABEL_66;
          ++v52;
        }
LABEL_73:
        if ( v53 )
          v44 = v53;
        goto LABEL_66;
      }
    }
    else
    {
      if ( (int)v4 - *(_DWORD *)(a1 + 244) - v46 > (unsigned int)v4 >> 3 )
      {
LABEL_66:
        *(_DWORD *)(a1 + 240) = v46;
        if ( *v44 != -8 )
          --*(_DWORD *)(a1 + 244);
        *v44 = a2;
        v44[1] = (__int64)(v44 + 3);
        v44[2] = 0x100000000LL;
        return;
      }
      sub_1CA0D00(a1 + 224, v4);
      v54 = *(_DWORD *)(a1 + 248);
      if ( v54 )
      {
        v55 = v54 - 1;
        v56 = *(_QWORD *)(a1 + 232);
        v57 = 1;
        v58 = v55 & v9;
        v44 = (__int64 *)(v56 + 48LL * v58);
        v46 = *(_DWORD *)(a1 + 240) + 1;
        v53 = 0;
        v59 = *v44;
        if ( *v44 == a2 )
          goto LABEL_66;
        while ( v59 != -8 )
        {
          if ( !v53 && v59 == -16 )
            v53 = v44;
          v58 = v55 & (v57 + v58);
          v44 = (__int64 *)(v56 + 48LL * v58);
          v59 = *v44;
          if ( *v44 == a2 )
            goto LABEL_66;
          ++v57;
        }
        goto LABEL_73;
      }
    }
    ++*(_DWORD *)(a1 + 240);
    BUG();
  }
  if ( v12 != (__int64 *)(48 * v4 + v6) )
  {
LABEL_4:
    v15 = (__int64 *)v12[1];
    v64 = &v15[3 * *((unsigned int *)v12 + 4)];
    if ( v15 != v64 )
    {
      for ( i = (__int64 *)v12[1]; v64 != i; i += 3 )
      {
        v17 = i[1];
        v74 = v76;
        v76[1] = v17;
        v76[0] = a3;
        v75 = 0x300000002LL;
        if ( i[2] )
        {
          v76[2] = i[2];
          LODWORD(v75) = 3;
        }
        if ( !sub_15CCEE0(*(_QWORD *)(a1 + 360), a3, *i) )
          goto LABEL_43;
        v18 = sub_15F3430(*i);
        v19 = sub_16498A0(v18);
        v88 = 0;
        v89 = 0;
        v20 = *(unsigned __int8 **)(v18 + 48);
        v85 = v19;
        v87 = 0;
        v21 = *(_QWORD *)(v18 + 40);
        v82 = 0;
        v83 = v21;
        v86 = 0;
        v84 = (__int64 *)(v18 + 24);
        v77[0] = v20;
        if ( v20 )
        {
          sub_1623A60((__int64)v77, (__int64)v20, 2);
          if ( v82 )
            sub_161E7C0((__int64)&v82, (__int64)v82);
          v82 = v77[0];
          if ( v77[0] )
            sub_1623210((__int64)v77, v77[0], (__int64)&v82);
        }
        strcpy(v73.m128i_i8, "align");
        v68 = 257;
        v22 = 8LL * (unsigned int)v75;
        v23 = _mm_load_si128(&v73);
        v71 = &v73;
        v77[0] = (unsigned __int8 *)&v78;
        v77[1] = (unsigned __int8 *)5;
        v72 = 0;
        v73.m128i_i8[0] = 0;
        v79 = 0;
        v80 = 0;
        v81 = 0;
        v78 = v23;
        if ( v22 )
        {
          srcb = v74;
          v24 = (char *)sub_22077B0(8LL * (unsigned int)v75);
          v25 = &v24[v22];
          v79 = v24;
          v81 = &v24[v22];
          memcpy(v24, srcb, v22);
        }
        else
        {
          v25 = 0;
        }
        v80 = v25;
        v26 = (__int64 *)sub_16498A0(*i);
        v66 = sub_159C4F0(v26);
        v27 = *(void **)(*i - 24);
        v70 = 257;
        src = v27;
        v28 = *(_QWORD *)(*(_QWORD *)v27 + 24LL);
        v29 = sub_1648AB0(72, (unsigned int)((v80 - v79) >> 3) + 2, 0x10u);
        v30 = (__int64)v29;
        if ( v29 )
        {
          v60 = (__int64)src;
          srca = v29;
          sub_15F1EA0(
            (__int64)v29,
            **(_QWORD **)(v28 + 16),
            54,
            (__int64)&v29[-3 * (unsigned int)((v80 - v79) >> 3) - 6],
            ((v80 - v79) >> 3) + 2,
            0);
          *(_QWORD *)(v30 + 56) = 0;
          sub_15F5B40(v30, v28, v60, &v66, 1, (__int64)v69, (__int64 *)v77, 1);
        }
        else
        {
          srca = 0;
        }
        v31 = *(_BYTE *)(*(_QWORD *)v30 + 8LL);
        if ( v31 == 16 )
          v31 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)v30 + 16LL) + 8LL);
        if ( (unsigned __int8)(v31 - 1) <= 5u || *(_BYTE *)(v30 + 16) == 76 )
        {
          v32 = v87;
          if ( v86 )
            sub_1625C10(v30, 3, v86);
          sub_15F2440(v30, v32);
        }
        if ( v83 )
        {
          v33 = v84;
          sub_157E9D0(v83 + 40, v30);
          v34 = *(_QWORD *)(v30 + 24);
          v35 = *v33;
          *(_QWORD *)(v30 + 32) = v33;
          v35 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v30 + 24) = v35 | v34 & 7;
          *(_QWORD *)(v35 + 8) = v30 + 24;
          *v33 = *v33 & 7 | (v30 + 24);
        }
        v36 = (unsigned __int8 **)(v30 + 48);
        sub_164B780((__int64)srca, &v67);
        if ( v82 )
        {
          v69[0] = v82;
          sub_1623A60((__int64)v69, (__int64)v82, 2);
          v37 = *(_QWORD *)(v30 + 48);
          if ( v37 )
            sub_161E7C0(v30 + 48, v37);
          v38 = (unsigned __int8 *)v69[0];
          *(_QWORD *)(v30 + 48) = v69[0];
          if ( v38 )
            sub_1623210((__int64)v69, v38, v30 + 48);
        }
        if ( v79 )
          j_j___libc_free_0(v79, v81 - v79);
        if ( (__m128i *)v77[0] != &v78 )
          j_j___libc_free_0(v77[0], v78.m128i_i64[0] + 1);
        if ( v71 != &v73 )
          j_j___libc_free_0(v71, v73.m128i_i64[0] + 1);
        v39 = *(unsigned __int8 **)(*i + 48);
        v77[0] = v39;
        if ( v39 )
        {
          sub_1623A60((__int64)v77, (__int64)v39, 2);
          if ( v36 != v77 )
          {
            v40 = *(_QWORD *)(v30 + 48);
            if ( v40 )
LABEL_51:
              sub_161E7C0(v30 + 48, v40);
            v41 = v77[0];
            *(unsigned __int8 **)(v30 + 48) = v77[0];
            if ( v41 )
              sub_1623210((__int64)v77, v41, v30 + 48);
            goto LABEL_41;
          }
          if ( v77[0] )
            sub_161E7C0(v30 + 48, (__int64)v77[0]);
        }
        else if ( v36 != v77 )
        {
          v40 = *(_QWORD *)(v30 + 48);
          if ( v40 )
            goto LABEL_51;
        }
LABEL_41:
        *(_QWORD *)(v30 + 56) = *(_QWORD *)(*i + 56);
        sub_14CE830(*(_QWORD *)(a1 + 368), v30);
        if ( v82 )
          sub_161E7C0((__int64)&v82, (__int64)v82);
LABEL_43:
        if ( v74 != v76 )
          _libc_free((unsigned __int64)v74);
      }
    }
  }
}
