// Function: sub_1FEABF0
// Address: 0x1feabf0
//
void __fastcall sub_1FEABF0(__int64 *a1, unsigned __int64 a2, char a3, unsigned __int8 a4, __int64 a5)
{
  __int64 *v5; // r15
  unsigned __int64 v7; // r12
  unsigned int v8; // ecx
  __int64 v9; // rdx
  unsigned __int64 v11; // rsi
  int v12; // ecx
  char v13; // r11
  __int64 v14; // rsi
  __int64 v15; // rdi
  __int64 (*v16)(); // rax
  unsigned int v17; // ecx
  unsigned int v18; // edx
  __int64 v19; // rsi
  char v20; // al
  __int64 v21; // rcx
  __int64 v22; // rsi
  __int16 v23; // ax
  __int64 v24; // rbx
  _QWORD *v25; // rax
  char v26; // r11
  unsigned int v27; // eax
  unsigned int v28; // r13d
  unsigned __int64 *v29; // rax
  unsigned int v30; // r8d
  unsigned __int16 v31; // ax
  __int64 v32; // rax
  __int64 v33; // rbx
  __int64 v34; // rdx
  __int64 *v35; // r13
  int v36; // r8d
  __int64 v37; // r9
  __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 v40; // rbx
  unsigned int v41; // r15d
  unsigned __int16 v42; // r13
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // rax
  __int64 v47; // r13
  __int64 v48; // r12
  int v49; // ecx
  __int16 v50; // ax
  __int64 v51; // rsi
  _WORD *v52; // rbx
  unsigned int v53; // eax
  __int64 v54; // r10
  __int64 v55; // rcx
  unsigned __int64 i; // rax
  __int64 v57; // rax
  __int64 v58; // rbx
  __int64 v59; // rax
  int v60; // eax
  __int64 v61; // rcx
  int v62; // ebx
  char v63; // r8
  __int64 v64; // rax
  char v65; // r9
  bool v66; // di
  bool v67; // si
  bool v68; // cl
  bool v69; // r10
  bool v70; // r8
  __int64 v71; // rax
  unsigned __int16 *v72; // [rsp+0h] [rbp-B0h]
  bool v74; // [rsp+Ah] [rbp-A6h]
  unsigned int v76; // [rsp+Ch] [rbp-A4h]
  __int64 v77; // [rsp+10h] [rbp-A0h]
  unsigned int v78; // [rsp+18h] [rbp-98h]
  unsigned int v79; // [rsp+1Ch] [rbp-94h]
  unsigned int v80; // [rsp+20h] [rbp-90h]
  char v81; // [rsp+28h] [rbp-88h]
  unsigned __int8 v82; // [rsp+28h] [rbp-88h]
  unsigned __int64 v83; // [rsp+28h] [rbp-88h]
  unsigned int v84; // [rsp+30h] [rbp-80h]
  unsigned __int64 v85; // [rsp+30h] [rbp-80h]
  __int64 *v86; // [rsp+30h] [rbp-80h]
  __int64 v87; // [rsp+30h] [rbp-80h]
  __int64 v88; // [rsp+30h] [rbp-80h]
  char v89; // [rsp+30h] [rbp-80h]
  __int64 v91; // [rsp+38h] [rbp-78h]
  int v92; // [rsp+38h] [rbp-78h]
  __int64 v93; // [rsp+40h] [rbp-70h] BYREF
  _QWORD *v94; // [rsp+48h] [rbp-68h]
  __m128i v95; // [rsp+50h] [rbp-60h] BYREF
  _QWORD v96[10]; // [rsp+60h] [rbp-50h] BYREF

  v5 = a1;
  v7 = a2;
  v8 = ~*(__int16 *)(a2 + 24);
  v74 = v8 == 10 || (unsigned int)(-*(__int16 *)(a2 + 24) - 8) <= 1;
  if ( v74 )
  {
    sub_1FE7FB0(a1, a2, a5, a3, a4);
    return;
  }
  if ( v8 == 11 )
  {
    sub_1FE8EF0((size_t *)a1, a2, a5);
    return;
  }
  if ( v8 == 14 )
  {
    sub_1FE92A0(a1, a2, a5, a3, a4);
    return;
  }
  if ( v8 != 9 )
  {
    v9 = a1[2];
    v11 = *(_QWORD *)(v9 + 8) + ((unsigned __int64)v8 << 6);
    v77 = v11;
    v72 = 0;
    v78 = sub_1FE6580(a2);
    v76 = *(unsigned __int8 *)(v11 + 4);
    if ( ((v12 - 19) & 0xFFFFFFFD) == 0 )
    {
      v14 = 13;
      if ( v12 == 21 )
      {
        v46 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v7 + 32) + 160LL) + 88LL);
        v14 = *(_QWORD *)(v46 + 24);
        if ( *(_DWORD *)(v46 + 32) > 0x40u )
          v14 = *(_QWORD *)v14;
        v76 = v78;
      }
      v72 = 0;
      v15 = v5[4];
      v16 = *(__int64 (**)())(*(_QWORD *)v15 + 1280LL);
      if ( v16 != sub_1FD3440 )
      {
        v89 = v13;
        v71 = ((__int64 (__fastcall *)(__int64, __int64))v16)(v15, v14);
        v13 = v89;
        v72 = (unsigned __int16 *)v71;
      }
    }
    v17 = *(_DWORD *)(v7 + 56);
    v18 = *(unsigned __int16 *)(v77 + 2) - v76;
    while ( 1 )
    {
      if ( !v17 )
      {
        v84 = 0;
        goto LABEL_19;
      }
      v19 = *(_QWORD *)(v7 + 32);
      v20 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v19 + 40LL * (v17 - 1)) + 40LL)
                     + 16LL * *(unsigned int *)(v19 + 40LL * (v17 - 1) + 8));
      if ( v20 != 111 )
        break;
      --v17;
    }
    if ( v20 == 1 )
      --v17;
    v84 = v17;
    if ( v18 < v17 )
    {
      v21 = v19 + 40LL * (v17 - 1);
      v22 = v21 - 40 - 40LL * (~v18 + v84);
      do
      {
        v23 = *(_WORD *)(*(_QWORD *)v21 + 24LL);
        if ( v23 != 9 && (v23 != 8 || *(int *)(*(_QWORD *)v21 + 84LL) <= 0) )
          break;
        v21 -= 40;
      }
      while ( v22 != v21 );
    }
LABEL_19:
    if ( v76 < v78 )
      v74 = *(_QWORD *)(v77 + 32) != 0;
    v24 = *v5;
    v81 = v13;
    v25 = sub_1E0B640(*v5, v77, (__int64 *)(v7 + 72), 0);
    v93 = v24;
    v94 = v25;
    v26 = v81;
    if ( v78 )
    {
      sub_1FE8880(v5, v7, &v93, v77, v81, a4, a5);
      v63 = *(_BYTE *)(v7 + 81);
      v64 = (__int64)v94;
      v65 = *(_BYTE *)(v7 + 80) >> 7;
      v66 = (*(_BYTE *)(v7 + 80) & 0x10) != 0;
      v67 = (*(_BYTE *)(v7 + 80) & 0x20) != 0;
      v68 = (v63 & 2) != 0;
      v69 = (v63 & 4) != 0;
      v70 = (v63 & 8) != 0;
      v26 = v81;
      if ( (*(_BYTE *)(v7 + 80) & 0x40) != 0 )
        *((_WORD *)v94 + 23) |= 0x40u;
      if ( v65 )
        *(_WORD *)(v64 + 46) |= 0x80u;
      if ( v66 )
        *(_WORD *)(v64 + 46) |= 0x10u;
      if ( v67 )
        *(_WORD *)(v64 + 46) |= 0x20u;
      if ( v68 )
        *(_WORD *)(v64 + 46) |= 0x100u;
      if ( v69 )
        *(_WORD *)(v64 + 46) |= 0x200u;
      if ( v70 )
        *(_WORD *)(v64 + 46) |= 0x400u;
    }
    v27 = 0;
    if ( v76 > v78 )
      v27 = v76 - v78;
    if ( v27 != v84 )
    {
      v82 = v26;
      v79 = a4;
      v28 = v27;
      v80 = v76 - v27;
      do
      {
        v29 = (unsigned __int64 *)(*(_QWORD *)(v7 + 32) + 40LL * v28);
        v30 = v28 + v80;
        ++v28;
        sub_1FE6BA0(v5, &v93, *v29, v29[1], v30, v77, a5, 0, v82, v79);
      }
      while ( v84 != v28 );
    }
    if ( v72 )
    {
      v31 = *v72;
      if ( *v72 )
      {
        v85 = v7;
        LODWORD(v7) = 0;
        do
        {
          v95.m128i_i32[2] = v31;
          memset(v96, 0, 24);
          v95.m128i_i64[0] = 0x430000000LL;
          sub_1E1A9C0((__int64)v94, v93, &v95);
          v7 = (unsigned int)(v7 + 1);
          v31 = v72[v7];
        }
        while ( v31 );
        v7 = v85;
      }
    }
    v32 = *(_QWORD *)(v7 + 88);
    v33 = (__int64)v94;
    v34 = (*(_QWORD *)(v7 + 96) - v32) >> 3;
    v94[7] = v32;
    *(_BYTE *)(v33 + 49) = v34;
    v35 = (__int64 *)v5[6];
    sub_1DD5BA0((__int64 *)(v5[5] + 16), v33);
    v38 = *v35;
    v39 = *(_QWORD *)v33;
    *(_QWORD *)(v33 + 8) = v35;
    v38 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v33 = v38 | v39 & 7;
    *(_QWORD *)(v38 + 8) = v33;
    *v35 = v33 | *v35 & 7;
    v95.m128i_i64[0] = (__int64)v96;
    v95.m128i_i64[1] = 0x800000000LL;
    if ( v74 && v76 < v78 )
    {
      v40 = 0;
      v86 = v5;
      v41 = v76;
      do
      {
        v42 = *(_WORD *)(*(_QWORD *)(v77 + 32) + v40);
        if ( (unsigned __int8)sub_1D18C40(v7, v41) )
        {
          v43 = v95.m128i_u32[2];
          if ( v95.m128i_i32[2] >= (unsigned __int32)v95.m128i_i32[3] )
          {
            sub_16CD150((__int64)&v95, v96, 0, 4, v36, v37);
            v43 = v95.m128i_u32[2];
          }
          *(_DWORD *)(v95.m128i_i64[0] + 4 * v43) = v42;
          ++v95.m128i_i32[2];
          sub_1FE9790(v86, v7, v41, a3, a4, v42, a5);
        }
        ++v41;
        v40 += 2;
      }
      while ( v78 != v41 );
      v5 = v86;
      v44 = v95.m128i_u32[2];
      if ( *(_BYTE *)(*(_QWORD *)(v7 + 40) + 16LL * (unsigned int)(*(_DWORD *)(v7 + 60) - 1)) != 111 )
        goto LABEL_41;
      v45 = *(_QWORD *)(v7 + 48);
      if ( !v45 )
        goto LABEL_41;
LABEL_55:
      while ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v45 + 40LL) + 16LL * *(unsigned int *)(v45 + 8)) != 111 )
      {
        v45 = *(_QWORD *)(v45 + 32);
        if ( !v45 )
          goto LABEL_41;
      }
      v47 = *(_QWORD *)(v45 + 16);
      if ( !v47 )
      {
LABEL_41:
        if ( (_DWORD)v44 )
          goto LABEL_44;
        goto LABEL_42;
      }
      v83 = v7;
      v48 = *(_QWORD *)(v45 + 16);
      v49 = *(unsigned __int16 *)(v47 + 24);
      v50 = v49;
      if ( (_WORD)v49 == 47 )
        goto LABEL_92;
LABEL_66:
      if ( v49 == 46 )
        goto LABEL_85;
      v51 = (unsigned int)v44;
      v52 = *(_WORD **)(*(_QWORD *)(v5[2] + 8) + ((unsigned __int64)(unsigned int)~v50 << 6) + 24);
      if ( !v52 || !*v52 )
        goto LABEL_95;
      v53 = 0;
      do
        v37 = 2LL * ++v53;
      while ( *(_WORD *)((char *)v52 + v37) );
      v54 = v37 >> 1;
      if ( v37 >> 1 > v95.m128i_u32[3] - (unsigned __int64)(unsigned int)v44 )
      {
        v87 = 2LL * v53;
        v91 = v37 >> 1;
        sub_16CD150((__int64)&v95, v96, v54 + (unsigned int)v44, 4, v36, v37);
        v51 = v95.m128i_u32[2];
        v37 = v87;
        LODWORD(v54) = v91;
      }
      v55 = v95.m128i_i64[0] + 4 * v51;
      if ( v37 )
      {
        for ( i = 0; i != v37; i += 2LL )
          *(_DWORD *)(v55 + 2 * i) = (unsigned __int16)v52[i / 2];
        LODWORD(v51) = v95.m128i_i32[2];
      }
      else
      {
LABEL_95:
        LODWORD(v54) = 0;
      }
      v95.m128i_i32[2] = v54 + v51;
      v57 = *(unsigned int *)(v48 + 56);
      v44 = (unsigned int)(v54 + v51);
      if ( !(_DWORD)v57 )
        goto LABEL_85;
      v58 = 0;
      v37 = 40 * v57;
      do
      {
        while ( 1 )
        {
          v59 = *(_QWORD *)(*(_QWORD *)(v48 + 32) + v58);
          if ( *(_WORD *)(v59 + 24) != 8 )
            break;
          v60 = *(_DWORD *)(v59 + 84);
          if ( v60 <= 0 )
            break;
          if ( v95.m128i_i32[3] <= (unsigned int)v44 )
          {
            v88 = v37;
            v92 = v60;
            sub_16CD150((__int64)&v95, v96, 0, 4, v36, v37);
            v44 = v95.m128i_u32[2];
            v37 = v88;
            v60 = v92;
          }
          v58 += 40;
          *(_DWORD *)(v95.m128i_i64[0] + 4 * v44) = v60;
          v44 = (unsigned int)++v95.m128i_i32[2];
          if ( v37 == v58 )
            goto LABEL_85;
        }
        v58 += 40;
      }
      while ( v37 != v58 );
LABEL_85:
      while ( 1 )
      {
        v61 = *(_QWORD *)(v48 + 48);
        if ( !v61 )
          break;
        while ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v61 + 40LL) + 16LL * *(unsigned int *)(v61 + 8)) != 111 )
        {
          v61 = *(_QWORD *)(v61 + 32);
          if ( !v61 )
            goto LABEL_88;
        }
        v48 = *(_QWORD *)(v61 + 16);
        if ( !v48 )
          break;
        v49 = *(unsigned __int16 *)(v48 + 24);
        v50 = *(_WORD *)(v48 + 24);
        if ( (_WORD)v49 != 47 )
          goto LABEL_66;
LABEL_92:
        v62 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v48 + 32) + 40LL) + 84LL);
        if ( v95.m128i_i32[3] <= (unsigned int)v44 )
        {
          sub_16CD150((__int64)&v95, v96, 0, 4, v36, v37);
          v44 = v95.m128i_u32[2];
        }
        *(_DWORD *)(v95.m128i_i64[0] + 4 * v44) = v62;
        v44 = (unsigned int)++v95.m128i_i32[2];
      }
LABEL_88:
      v7 = v83;
      if ( (_DWORD)v44 )
        goto LABEL_44;
    }
    else if ( *(_BYTE *)(*(_QWORD *)(v7 + 40) + 16LL * (unsigned int)(*(_DWORD *)(v7 + 60) - 1)) == 111 )
    {
      v45 = *(_QWORD *)(v7 + 48);
      v44 = 0;
      if ( v45 )
        goto LABEL_55;
    }
LABEL_42:
    if ( !*(_QWORD *)(v77 + 32) )
    {
LABEL_45:
      if ( (*(_BYTE *)(v77 + 11) & 1) != 0 )
        (*(void (__fastcall **)(__int64, _QWORD *, unsigned __int64))(*(_QWORD *)v5[4] + 1464LL))(v5[4], v94, v7);
      if ( (_QWORD *)v95.m128i_i64[0] != v96 )
        _libc_free(v95.m128i_u64[0]);
      return;
    }
    v44 = 0;
LABEL_44:
    sub_1E1B900((__int64)v94, (int *)v95.m128i_i64[0], v44, v5[3]);
    goto LABEL_45;
  }
}
