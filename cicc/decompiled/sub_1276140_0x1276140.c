// Function: sub_1276140
// Address: 0x1276140
//
_QWORD *__fastcall sub_1276140(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // r12
  __int64 v4; // r13
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // rax
  const __m128i *v8; // r14
  __int64 v9; // rcx
  __int64 v10; // r8
  int v11; // edx
  int v12; // edx
  __int64 v13; // rsi
  __int64 *v14; // rax
  unsigned __int64 v15; // rsi
  const __m128i *v16; // r13
  __int64 i; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  char v20; // al
  int v21; // eax
  _QWORD *v22; // rbx
  _QWORD *v23; // r14
  __int64 v24; // rdi
  _QWORD *v25; // rbx
  _QWORD *v26; // r14
  char *v28; // rax
  __int64 v29; // rbx
  unsigned int v30; // r14d
  __int64 v31; // rax
  __int64 v32; // r12
  char *v33; // r15
  char *v34; // rdx
  __int64 v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // r15
  __int64 v38; // rdx
  _QWORD *v39; // rax
  _QWORD *v40; // r14
  unsigned __int64 v41; // r15
  __int64 v42; // rcx
  int v43; // r10d
  _QWORD *v44; // rdx
  unsigned int v45; // r8d
  __int64 v46; // r15
  __int64 v47; // rax
  char *v48; // r14
  unsigned __int16 v49; // cx
  char v50; // dl
  __int64 v51; // rcx
  __int64 v52; // rdx
  char v53; // al
  int v54; // eax
  int v55; // edi
  int v56; // [rsp+8h] [rbp-268h]
  unsigned int v57; // [rsp+14h] [rbp-25Ch]
  unsigned __int64 v58; // [rsp+18h] [rbp-258h]
  _QWORD *v59; // [rsp+18h] [rbp-258h]
  __int64 v60; // [rsp+20h] [rbp-250h]
  __int64 v61; // [rsp+28h] [rbp-248h]
  const __m128i *v62; // [rsp+30h] [rbp-240h]
  __int64 v63; // [rsp+50h] [rbp-220h]
  __int64 v64; // [rsp+58h] [rbp-218h]
  char v65; // [rsp+60h] [rbp-210h] BYREF
  __int16 v66; // [rsp+70h] [rbp-200h]
  _BYTE *v67; // [rsp+80h] [rbp-1F0h] BYREF
  __int64 v68; // [rsp+88h] [rbp-1E8h]
  _BYTE v69[32]; // [rsp+90h] [rbp-1E0h] BYREF
  _QWORD v70[37]; // [rsp+B0h] [rbp-1C0h] BYREF
  _QWORD *v71; // [rsp+1D8h] [rbp-98h]
  _QWORD *v72; // [rsp+1E0h] [rbp-90h]
  __int64 v73; // [rsp+1E8h] [rbp-88h]
  __int64 v74; // [rsp+200h] [rbp-70h]
  _QWORD *v75; // [rsp+220h] [rbp-50h]
  unsigned int v76; // [rsp+230h] [rbp-40h]

  v2 = a1;
  v3 = a2;
  qword_4F04C50 = sub_72B840(a2);
  v4 = sub_1299230(a1 + 8, a2);
  v7 = sub_1276020(a1, a2, v4, v5, v6);
  v8 = (const __m128i *)v7;
  if ( *(_BYTE *)(v7 + 16) == 5 )
  {
    if ( *(_WORD *)(v7 + 18) != 47 )
      sub_127B550("unexpected error in codegen for function!");
    v8 = *(const __m128i **)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF));
  }
  if ( v4 != v8[1].m128i_i64[1] )
  {
    if ( !(unsigned __int8)sub_15E4F60(v8) )
      sub_127B550("unexpected error in codegen for function: found previous definition of same function!");
    v11 = *(_DWORD *)(a1 + 416);
    if ( v11 )
    {
      v12 = v11 - 1;
      v13 = *(_QWORD *)(a1 + 400);
      v9 = v12 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v14 = (__int64 *)(v13 + 16 * v9);
      v10 = *v14;
      if ( v3 == *v14 )
      {
LABEL_8:
        *v14 = -8;
        --*(_DWORD *)(v2 + 408);
        ++*(_DWORD *)(v2 + 412);
      }
      else
      {
        v54 = 1;
        while ( v10 != -4 )
        {
          v55 = v54 + 1;
          v9 = v12 & (unsigned int)(v54 + v9);
          v14 = (__int64 *)(v13 + 16LL * (unsigned int)v9);
          v10 = *v14;
          if ( v3 == *v14 )
            goto LABEL_8;
          v54 = v55;
        }
      }
    }
    v15 = (unsigned __int64)v8;
    v64 = sub_1276020(v2, v3, v4, v9, v10);
    v16 = (const __m128i *)v64;
    sub_164B7C0(v64, v8);
    for ( i = *(_QWORD *)(v3 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v18 = *(_QWORD *)(i + 168);
    if ( v18 && (*(_BYTE *)(v18 + 16) & 2) == 0 )
    {
      if ( !v8[1].m128i_i8[0] )
      {
        v63 = **(_QWORD **)(*(_QWORD *)(v64 + 24) + 16LL);
        v67 = v69;
        v68 = 0x400000000LL;
        if ( v8->m128i_i64[1] )
        {
          v62 = v8;
          v29 = v8->m128i_i64[1];
          v61 = v2;
          v60 = v3;
          do
          {
            v30 = sub_1648720(v29);
            v31 = sub_1648700(v29);
            v29 = *(_QWORD *)(v29 + 8);
            v32 = v31;
            if ( *(_BYTE *)(v31 + 16) == 78 && !v30 && (v63 == *(_QWORD *)v31 || !*(_QWORD *)(v31 + 8)) )
            {
              if ( (*(_BYTE *)(v64 + 18) & 1) != 0 )
              {
                sub_15E08E0(v64);
                v33 = *(char **)(v64 + 88);
                if ( (*(_BYTE *)(v64 + 18) & 1) != 0 )
                  sub_15E08E0(v64);
                v34 = *(char **)(v64 + 88);
              }
              else
              {
                v33 = *(char **)(v64 + 88);
                v34 = v33;
              }
              v15 = (unsigned __int64)&v34[40 * *(_QWORD *)(v64 + 96)];
              v35 = *(_DWORD *)(v32 + 20) & 0xFFFFFFF;
              if ( (char *)v15 == v33 )
              {
                v36 = 0;
LABEL_62:
                v37 = 3 * (1 - v35 + v36);
                v38 = 24 * (1 - v35);
                v37 *= 8;
                v39 = (_QWORD *)(v32 + v37);
                v40 = (_QWORD *)(v32 + v38);
                v41 = 0xAAAAAAAAAAAAAAABLL * ((v37 - v38) >> 3);
                v42 = (unsigned int)v68;
                if ( v41 > HIDWORD(v68) - (unsigned __int64)(unsigned int)v68 )
                {
                  v59 = v39;
                  sub_16CD150(&v67, v69, v41 + (unsigned int)v68, 8);
                  v42 = (unsigned int)v68;
                  v39 = v59;
                }
                v43 = (int)v67;
                v44 = &v67[8 * v42];
                if ( v39 != v40 )
                {
                  do
                  {
                    if ( v44 )
                      *v44 = *v40;
                    v40 += 3;
                    ++v44;
                  }
                  while ( v39 != v40 );
                  v43 = (int)v67;
                  LODWORD(v42) = v68;
                }
                LODWORD(v68) = v41 + v42;
                v45 = v41 + v42 + 1;
                v46 = (unsigned int)(v41 + v42);
                v66 = 257;
                v15 = v45;
                v56 = v43;
                v57 = v68 + 1;
                v58 = *(_QWORD *)(*(_QWORD *)v64 + 24LL);
                v47 = sub_1648AB0(72, v45, 0);
                v48 = (char *)v47;
                if ( v47 )
                {
                  sub_15F1EA0(v47, **(_QWORD **)(v58 + 16), 54, v47 - 24 * v46 - 24, v57, v32);
                  *((_QWORD *)v48 + 7) = 0;
                  v15 = v58;
                  sub_15F5B40((_DWORD)v48, v58, v64, v56, v46, (unsigned int)&v65, 0, 0);
                }
                LODWORD(v68) = 0;
                if ( *(_BYTE *)(*(_QWORD *)v48 + 8LL) )
                {
                  v15 = v32;
                  sub_164B7C0(v48, v32);
                }
                v49 = *((_WORD *)v48 + 9);
                *((_QWORD *)v48 + 7) = *(_QWORD *)(v32 + 56);
                v50 = v49;
                v51 = v49 & 0x8000;
                v52 = v50 & 3;
                *((_WORD *)v48 + 9) = v51 | v52 | (4 * ((*(_WORD *)(v32 + 18) >> 2) & 0xDFFF));
                if ( *(_QWORD *)(v32 + 8) )
                {
                  v15 = (unsigned __int64)v48;
                  sub_164D160(v32, v48);
                }
                if ( *(_QWORD *)(v32 + 48) || *(__int16 *)(v32 + 18) < 0 )
                {
                  v15 = (unsigned __int64)"dbg";
                  v51 = sub_1625940(v32, "dbg", 3);
                  if ( v51 )
                  {
                    v15 = (unsigned __int64)"dbg";
                    sub_1626100(v48, "dbg", 3, v51);
                  }
                }
                sub_15F20C0(v32, v15, v52, v51);
              }
              else
              {
                while ( (*(_DWORD *)(v32 + 20) & 0xFFFFFFF) - 1 != v30 )
                {
                  v36 = ++v30;
                  if ( **(_QWORD **)(v32 + 24 * (v30 - v35)) != *(_QWORD *)v33 )
                    break;
                  v33 += 40;
                  if ( v33 == (char *)v15 )
                    goto LABEL_62;
                }
              }
            }
          }
          while ( v29 );
          v8 = v62;
          v2 = v61;
          v3 = v60;
          if ( v67 != v69 )
            _libc_free(v67, v15);
        }
      }
      sub_159D9E0(v8);
      if ( !v8->m128i_i64[1] )
      {
LABEL_15:
        sub_15E5B20(v8);
        v20 = sub_15E4F60(v64);
        if ( v20 )
          goto LABEL_16;
LABEL_46:
        v28 = sub_8258E0(v3, 0);
        return (_QWORD *)sub_6851A0(0xD83u, (_DWORD *)(v3 + 64), (__int64)v28);
      }
    }
    else if ( !v8->m128i_i64[1] )
    {
      goto LABEL_15;
    }
    v19 = sub_15A4510(v64, v8->m128i_i64[0], 0);
    sub_164D160(v8, v19);
    goto LABEL_15;
  }
  v16 = v8;
  v20 = sub_15E4F60(v8);
  if ( !v20 )
    goto LABEL_46;
LABEL_16:
  if ( (*(_BYTE *)(v3 - 8) & 0x10) != 0 )
  {
    v16[2].m128i_i8[0] &= 0xF0u;
    goto LABEL_42;
  }
  v21 = sub_1268C40(v3, dword_4D046B4 != 0);
  if ( v21 == 7 )
  {
    v16[2].m128i_i8[0] = v16[2].m128i_i8[0] & 0xC0 | 7;
    goto LABEL_20;
  }
  if ( v21 != 8 )
  {
    v53 = v21 & 0xF;
    v16[2].m128i_i8[0] = v53 | v16[2].m128i_i8[0] & 0xF0;
    if ( ((v53 + 9) & 0xFu) <= 1 )
      goto LABEL_20;
    v20 = v53 != 9;
LABEL_42:
    if ( (v16[2].m128i_i8[0] & 0x30) == 0 || !v20 )
      goto LABEL_21;
    goto LABEL_20;
  }
  v16[2].m128i_i8[0] = v16[2].m128i_i8[0] & 0xC0 | 8;
LABEL_20:
  v16[2].m128i_i8[1] |= 0x40u;
LABEL_21:
  sub_12A4B10(&v67, v2);
  sub_12A5570(&v67, v3, v16);
  if ( v76 )
  {
    v22 = v75;
    v23 = &v75[4 * v76];
    do
    {
      if ( *v22 != -8 && *v22 != -16 )
      {
        v24 = v22[1];
        if ( v24 )
          j_j___libc_free_0(v24, v22[3] - v24);
      }
      v22 += 4;
    }
    while ( v23 != v22 );
  }
  j___libc_free_0(v75);
  j___libc_free_0(v74);
  v25 = v72;
  v26 = v71;
  if ( v72 != v71 )
  {
    do
    {
      if ( *v26 )
        j_j___libc_free_0(*v26, v26[2] - *v26);
      v26 += 3;
    }
    while ( v25 != v26 );
    v26 = v71;
  }
  if ( v26 )
    j_j___libc_free_0(v26, v73 - (_QWORD)v26);
  sub_1269670(v70[33]);
  sub_12694A0(v70[27]);
  sub_1269670(v70[21]);
  if ( v70[0] )
    sub_161E7C0(v70);
  j___libc_free_0(v68);
  sub_1297BC0(v2, v3, v16);
  if ( (unsigned __int8)sub_127B330(v3) )
    sub_1273CD0(v2, (__int64)v16);
  sub_1273830((_QWORD *)v2, v16, v3);
  qword_4F04C50 = 0;
  return &qword_4F04C50;
}
