// Function: sub_9172F0
// Address: 0x9172f0
//
_QWORD *__fastcall sub_9172F0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // r12
  __int64 v4; // r14
  __int64 v5; // rax
  const __m128i *v6; // rbx
  int v7; // ecx
  __int64 v8; // rsi
  int v9; // ecx
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // r8
  unsigned __int64 v13; // rsi
  const __m128i *v14; // r14
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 i; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  char v21; // al
  int v22; // eax
  char *v24; // rax
  __int64 v25; // rbx
  unsigned int v26; // eax
  unsigned __int64 v27; // r12
  unsigned int v28; // r13d
  __int64 v29; // r15
  __int64 v30; // rdx
  __int64 v31; // rdi
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r13
  _QWORD *v35; // r15
  _QWORD *v36; // rax
  __int64 v37; // r13
  __int64 v38; // rcx
  __int64 v39; // r13
  unsigned __int64 v40; // rdx
  int v41; // r10d
  _QWORD *v42; // rdx
  int v43; // eax
  unsigned int v44; // ecx
  unsigned __int64 v45; // r15
  __int64 v46; // rcx
  __int64 v47; // r13
  __int64 v48; // r9
  __int64 v49; // rdx
  char v50; // al
  int v51; // eax
  int v52; // edi
  __int64 v53; // [rsp-8h] [rbp-2F8h]
  __int64 v54; // [rsp+8h] [rbp-2E8h]
  __int64 v55; // [rsp+10h] [rbp-2E0h]
  const __m128i *v56; // [rsp+18h] [rbp-2D8h]
  int v57; // [rsp+20h] [rbp-2D0h]
  __int64 v58; // [rsp+30h] [rbp-2C0h]
  unsigned int v59; // [rsp+38h] [rbp-2B8h]
  unsigned int v60; // [rsp+3Ch] [rbp-2B4h]
  int v61; // [rsp+40h] [rbp-2B0h]
  _QWORD *v62; // [rsp+40h] [rbp-2B0h]
  __int64 v63; // [rsp+60h] [rbp-290h]
  __int64 v64; // [rsp+68h] [rbp-288h]
  char v65; // [rsp+70h] [rbp-280h] BYREF
  __int16 v66; // [rsp+90h] [rbp-260h]
  _BYTE *v67; // [rsp+A0h] [rbp-250h] BYREF
  __int64 v68; // [rsp+A8h] [rbp-248h]
  _BYTE v69[576]; // [rsp+B0h] [rbp-240h] BYREF

  v2 = a1;
  v3 = a2;
  qword_4F04C50 = sub_72B840(a2);
  v4 = sub_939EC0(a1 + 8, a2);
  v5 = sub_917010(a1, a2, v4);
  v6 = (const __m128i *)v5;
  if ( *(_BYTE *)v5 == 5 )
  {
    if ( *(_WORD *)(v5 + 2) != 49 )
      sub_91B8A0("unexpected error in codegen for function!");
    v6 = *(const __m128i **)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF));
  }
  if ( v4 != v6[1].m128i_i64[1] )
  {
    if ( !(unsigned __int8)sub_B2FC80(v6) )
      sub_91B8A0("unexpected error in codegen for function: found previous definition of same function!");
    v7 = *(_DWORD *)(a1 + 400);
    v8 = *(_QWORD *)(a1 + 384);
    if ( v7 )
    {
      v9 = v7 - 1;
      v10 = v9 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v11 = (__int64 *)(v8 + 16LL * v10);
      v12 = *v11;
      if ( v3 == *v11 )
      {
LABEL_8:
        *v11 = -8192;
        --*(_DWORD *)(v2 + 392);
        ++*(_DWORD *)(v2 + 396);
      }
      else
      {
        v51 = 1;
        while ( v12 != -4096 )
        {
          v52 = v51 + 1;
          v10 = v9 & (v51 + v10);
          v11 = (__int64 *)(v8 + 16LL * v10);
          v12 = *v11;
          if ( v3 == *v11 )
            goto LABEL_8;
          v51 = v52;
        }
      }
    }
    v13 = (unsigned __int64)v6;
    v64 = sub_917010(v2, v3, v4);
    v14 = (const __m128i *)v64;
    sub_BD6B90(v64, v6);
    for ( i = *(_QWORD *)(v3 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v19 = *(_QWORD *)(i + 168);
    if ( v19 && (*(_BYTE *)(v19 + 16) & 2) == 0 )
    {
      if ( !v6->m128i_i8[0] )
      {
        v63 = **(_QWORD **)(*(_QWORD *)(v64 + 24) + 16LL);
        v67 = v69;
        v68 = 0x400000000LL;
        if ( v6[1].m128i_i64[0] )
        {
          v56 = v6;
          v25 = v6[1].m128i_i64[0];
          v55 = v2;
          v54 = v3;
          do
          {
            v26 = sub_BD2910(v25);
            v27 = *(_QWORD *)(v25 + 24);
            v25 = *(_QWORD *)(v25 + 8);
            v28 = v26;
            if ( *(_BYTE *)v27 == 85 && !v26 && (v63 == *(_QWORD *)(v27 + 8) || !*(_QWORD *)(v27 + 16)) )
            {
              if ( (*(_BYTE *)(v64 + 2) & 1) != 0 )
              {
                sub_B2C6D0(v64);
                v29 = *(_QWORD *)(v64 + 96);
                if ( (*(_BYTE *)(v64 + 2) & 1) != 0 )
                  sub_B2C6D0(v64);
                v30 = *(_QWORD *)(v64 + 96);
              }
              else
              {
                v29 = *(_QWORD *)(v64 + 96);
                v30 = v29;
              }
              v13 = v30 + 40LL * *(_QWORD *)(v64 + 104);
              v31 = *(_DWORD *)(v27 + 4) & 0x7FFFFFF;
              if ( v13 == v29 )
              {
                v32 = 0;
LABEL_46:
                v33 = 32 * (1 - v31);
                v34 = 32 * (1 - v31 + v32);
                v35 = (_QWORD *)(v27 + v33);
                v36 = (_QWORD *)(v27 + v34);
                v37 = v34 - v33;
                v38 = (unsigned int)v68;
                v39 = v37 >> 5;
                v40 = v39 + (unsigned int)v68;
                if ( v40 > HIDWORD(v68) )
                {
                  v62 = v36;
                  sub_C8D5F0(&v67, v69, v40, 8);
                  v38 = (unsigned int)v68;
                  v36 = v62;
                }
                v41 = (int)v67;
                v42 = &v67[8 * v38];
                if ( v36 != v35 )
                {
                  do
                  {
                    if ( v42 )
                      *v42 = *v35;
                    v35 += 4;
                    ++v42;
                  }
                  while ( v36 != v35 );
                  v41 = (int)v67;
                  LODWORD(v38) = v68;
                }
                LODWORD(v68) = v39 + v38;
                v43 = v39 + v38;
                v44 = v39 + v38 + 1;
                v66 = 257;
                v13 = v44;
                v45 = *(_QWORD *)(v64 + 24);
                v57 = v41;
                v61 = v43;
                v59 = v44;
                v47 = sub_BD2C40(88, v44);
                if ( v47 )
                {
                  v48 = v58;
                  LOWORD(v48) = 0;
                  v60 = v59 & 0x7FFFFFF | v60 & 0xE0000000;
                  sub_B44260(v47, **(_QWORD **)(v45 + 16), 56, v60, v27 + 24, v48);
                  *(_QWORD *)(v47 + 72) = 0;
                  v13 = v45;
                  sub_B4A290(v47, v45, v64, v57, v61, (unsigned int)&v65, 0, 0);
                  v46 = v53;
                }
                LODWORD(v68) = 0;
                if ( *(_BYTE *)(*(_QWORD *)(v47 + 8) + 8LL) != 7 )
                {
                  v13 = v27;
                  sub_BD6B90(v47, v27);
                }
                *(_QWORD *)(v47 + 72) = *(_QWORD *)(v27 + 72);
                v49 = *(_WORD *)(v27 + 2) & 0xFFC;
                *(_WORD *)(v47 + 2) = v49 | *(_WORD *)(v47 + 2) & 0xF003;
                if ( *(_QWORD *)(v27 + 16) )
                {
                  v13 = v47;
                  sub_BD84D0(v27, v47);
                }
                if ( *(_QWORD *)(v27 + 48) || (*(_BYTE *)(v27 + 7) & 0x20) != 0 )
                {
                  v13 = (unsigned __int64)"dbg";
                  v46 = sub_B91F50(v27, "dbg", 3);
                  if ( v46 )
                  {
                    v13 = (unsigned __int64)"dbg";
                    sub_B9A090(v47, "dbg", 3, v46);
                  }
                }
                sub_B43D60(v27, v13, v49, v46);
              }
              else
              {
                while ( v28 != (*(_DWORD *)(v27 + 4) & 0x7FFFFFF) - 1 )
                {
                  v32 = ++v28;
                  if ( *(_QWORD *)(*(_QWORD *)(v27 + 32 * (v28 - v31)) + 8LL) != *(_QWORD *)(v29 + 8) )
                    break;
                  v29 += 40;
                  if ( v29 == v13 )
                    goto LABEL_46;
                }
              }
            }
          }
          while ( v25 );
          v6 = v56;
          v2 = v55;
          v3 = v54;
          if ( v67 != v69 )
            _libc_free(v67, v13);
        }
      }
      sub_AD0030(v6);
      if ( !v6[1].m128i_i64[0] )
      {
LABEL_15:
        sub_B30810(v6);
        v21 = sub_B2FC80(v64);
        if ( v21 )
          goto LABEL_16;
LABEL_30:
        v24 = sub_8258E0(v3, 0);
        return (_QWORD *)sub_6851A0(0xD83u, (_DWORD *)(v3 + 64), (__int64)v24);
      }
    }
    else if ( !v6[1].m128i_i64[0] )
    {
      goto LABEL_15;
    }
    v20 = sub_AD4C90(v64, v6->m128i_i64[1], 0, v15, v16, v17);
    sub_BD84D0(v6, v20);
    goto LABEL_15;
  }
  v14 = v6;
  v21 = sub_B2FC80(v6);
  if ( !v21 )
    goto LABEL_30;
LABEL_16:
  if ( (*(_BYTE *)(v3 - 8) & 0x10) != 0 )
  {
    v14[2].m128i_i8[0] &= 0xF0u;
  }
  else
  {
    v22 = sub_909290(v3, unk_4D046B4 != 0);
    if ( v22 == 7 )
    {
      v14[2].m128i_i16[0] = v14[2].m128i_i16[0] & 0xFCC0 | 7;
      goto LABEL_20;
    }
    if ( v22 == 8 )
    {
      v14[2].m128i_i16[0] = v14[2].m128i_i16[0] & 0xFCC0 | 8;
LABEL_20:
      v14[2].m128i_i8[1] |= 0x40u;
      goto LABEL_21;
    }
    v50 = v22 & 0xF;
    v14[2].m128i_i8[0] = v50 | v14[2].m128i_i8[0] & 0xF0;
    if ( ((v50 + 9) & 0xFu) <= 1 )
      goto LABEL_20;
    v21 = v50 != 9;
  }
  if ( (v14[2].m128i_i8[0] & 0x30) != 0 && v21 )
    goto LABEL_20;
LABEL_21:
  sub_9459E0(&v67, v2);
  sub_946910(&v67, v3, v14);
  sub_917130((__int64)&v67);
  sub_938140(v2, v3, v14);
  if ( (unsigned __int8)sub_91B680(v3) )
    sub_914140(v2, (__int64)v14);
  sub_913C80((_QWORD *)v2, v14, v3);
  qword_4F04C50 = 0;
  return &qword_4F04C50;
}
