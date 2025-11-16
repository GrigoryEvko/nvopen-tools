// Function: sub_2FE8000
// Address: 0x2fe8000
//
__int64 __fastcall sub_2FE8000(__int64 a1, __int64 a2)
{
  __int64 i; // rax
  bool v4; // zf
  __int64 v5; // rcx
  __int64 v6; // rsi
  __int64 v7; // rdx
  unsigned int v8; // eax
  __int16 v9; // r8
  __int64 v10; // rsi
  unsigned int v11; // edx
  __int16 v12; // di
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  __int16 v16; // r15
  __int64 j; // rbx
  __int64 v18; // r8
  __int64 (__fastcall *v19)(__int64, unsigned __int16); // rax
  unsigned __int16 v20; // r12
  unsigned int v21; // esi
  unsigned __int16 v22; // dx
  unsigned __int16 v23; // ax
  unsigned int v24; // r14d
  __int16 v25; // di
  __int16 v26; // bx
  int v27; // esi
  unsigned __int16 v28; // ax
  unsigned __int16 v29; // ax
  int v30; // edx
  int v31; // ecx
  unsigned __int16 v32; // si
  unsigned int v33; // r9d
  __int64 v34; // rsi
  __int64 v35; // rax
  unsigned __int64 v36; // rsi
  __int64 v37; // rdx
  unsigned __int64 v38; // rax
  _BYTE *v39; // rdx
  unsigned __int16 v40; // si
  unsigned int v41; // eax
  int v42; // esi
  __int16 v43; // ax
  char v44; // al
  _BYTE *v45; // rax
  signed int v46; // eax
  __int64 v47; // rdx
  unsigned int v48; // eax
  int v49; // esi
  __int64 v50; // rbx
  __int64 result; // rax
  char v52; // dl
  __int16 v53; // bx
  __int16 v54; // dx
  __int16 v55; // dx
  __int16 v56; // ax
  __int16 v57; // ax
  __int64 (*v58)(); // rdx
  __int16 v59; // dx
  __int16 v60; // dx
  __int16 v61; // dx
  __int64 (*v62)(); // rdx
  __int16 v63; // dx
  __int16 v64; // ax
  __int16 v65; // ax
  unsigned int v67; // [rsp+Ch] [rbp-94h]
  __int64 v68; // [rsp+10h] [rbp-90h]
  unsigned int v69; // [rsp+10h] [rbp-90h]
  unsigned int v70; // [rsp+18h] [rbp-88h]
  unsigned int v71; // [rsp+18h] [rbp-88h]
  char v72; // [rsp+1Ch] [rbp-84h]
  unsigned __int8 v73; // [rsp+21h] [rbp-7Fh]
  unsigned __int16 v74; // [rsp+22h] [rbp-7Eh]
  bool v75; // [rsp+24h] [rbp-7Ch]
  __int16 v76; // [rsp+24h] [rbp-7Ch]
  unsigned int v77; // [rsp+28h] [rbp-78h]
  unsigned int v78; // [rsp+2Ch] [rbp-74h]
  __int16 v79; // [rsp+30h] [rbp-70h] BYREF
  __int64 v80; // [rsp+38h] [rbp-68h]
  unsigned __int16 v81; // [rsp+40h] [rbp-60h] BYREF
  __int64 v82; // [rsp+48h] [rbp-58h]
  unsigned __int64 v83; // [rsp+50h] [rbp-50h]
  _BYTE *v84; // [rsp+58h] [rbp-48h]
  __int64 v85; // [rsp+60h] [rbp-40h]
  __int64 v86; // [rsp+68h] [rbp-38h]

  for ( i = 0; i != 274; ++i )
  {
    *(_WORD *)(a1 + 2 * i + 5866) = i;
    *(_WORD *)(a1 + 2 * i + 2304) = 1;
    *(_WORD *)(a1 + 2 * i + 2852) = i;
  }
  v4 = *(_QWORD *)(a1 + 184) == 0;
  v5 = 9;
  *(_WORD *)(a1 + 2830) = 0;
  if ( v4 )
  {
    do
    {
      v6 = (unsigned int)(v5 - 1);
      v7 = (unsigned int)v5;
      v5 = v6;
      v8 = v7 - 2;
    }
    while ( !*(_QWORD *)(a1 + 8 * v6 + 112) );
    v9 = v6;
    v10 = a1 + 2 * v7;
    v11 = v5;
    do
    {
      v10 += 2;
      v12 = *(_WORD *)(a1 + 2LL * v11 + 2304);
      *(_WORD *)(v10 + 5864) = v11++;
      *(_WORD *)(v10 + 2850) = v9;
      *(_WORD *)(v10 + 2302) = 2 * v12;
      *(_BYTE *)(a1 + (int)v11 + 524896) = 2;
    }
    while ( v11 != 9 );
    if ( v8 <= 1 )
      goto LABEL_13;
  }
  else
  {
    v8 = 8;
  }
  v10 = a1 + 2LL * v8;
  do
  {
    while ( 1 )
    {
      v13 = (unsigned __int16)v8;
      if ( !(_WORD)v8 || !*(_QWORD *)(a1 + 8LL * (unsigned __int16)v8 + 112) )
        break;
      v5 = v8--;
      v10 -= 2;
      if ( v8 == 1 )
        goto LABEL_13;
    }
    --v8;
    *(_WORD *)(v10 + 5866) = v5;
    v10 -= 2;
    *(_WORD *)(v10 + 2854) = v5;
    *(_BYTE *)(a1 + v13 + 524896) = 1;
  }
  while ( v8 != 1 );
LABEL_13:
  v14 = *(_QWORD *)(a1 + 216);
  if ( *(_QWORD *)(a1 + 240) )
    goto LABEL_14;
  if ( !v14 )
  {
    v61 = *(_WORD *)(a1 + 2322);
    v10 = 9;
    *(_BYTE *)(a1 + 524912) = 3;
    *(_WORD *)(a1 + 5898) = 9;
    *(_WORD *)(a1 + 2336) = v61;
    *(_WORD *)(a1 + 2884) = *(_WORD *)(a1 + 2870);
LABEL_14:
    if ( *(_QWORD *)(a1 + 232) )
      goto LABEL_15;
    goto LABEL_140;
  }
  v53 = *(_WORD *)(a1 + 2330);
  v5 = 13;
  v4 = *(_QWORD *)(a1 + 232) == 0;
  *(_BYTE *)(a1 + 524912) = 4;
  *(_WORD *)(a1 + 5898) = 13;
  *(_WORD *)(a1 + 2336) = 2 * v53;
  *(_WORD *)(a1 + 2884) = 13;
  if ( !v4 )
  {
LABEL_15:
    if ( *(_QWORD *)(a1 + 224) )
      goto LABEL_16;
    goto LABEL_141;
  }
LABEL_140:
  v54 = *(_WORD *)(a1 + 2322);
  v4 = *(_QWORD *)(a1 + 224) == 0;
  *(_BYTE *)(a1 + 524911) = 3;
  *(_WORD *)(a1 + 5896) = 9;
  *(_WORD *)(a1 + 2334) = v54;
  *(_WORD *)(a1 + 2882) = *(_WORD *)(a1 + 2870);
  if ( !v4 )
  {
LABEL_16:
    if ( v14 )
      goto LABEL_17;
    goto LABEL_142;
  }
LABEL_141:
  v55 = *(_WORD *)(a1 + 2318);
  *(_BYTE *)(a1 + 524910) = 3;
  *(_WORD *)(a1 + 5894) = 7;
  *(_WORD *)(a1 + 2332) = 3 * v55;
  *(_WORD *)(a1 + 2880) = *(_WORD *)(a1 + 2866);
  if ( v14 )
  {
LABEL_17:
    if ( *(_QWORD *)(a1 + 208) )
      goto LABEL_18;
    goto LABEL_143;
  }
LABEL_142:
  v56 = *(_WORD *)(a1 + 2320);
  v4 = *(_QWORD *)(a1 + 208) == 0;
  *(_BYTE *)(a1 + 524909) = 3;
  *(_WORD *)(a1 + 5892) = 8;
  *(_WORD *)(a1 + 2330) = v56;
  *(_WORD *)(a1 + 2878) = *(_WORD *)(a1 + 2868);
  if ( !v4 )
  {
LABEL_18:
    v15 = *(_QWORD *)a1;
    if ( *(_QWORD *)(a1 + 200) )
      goto LABEL_19;
    goto LABEL_144;
  }
LABEL_143:
  v57 = *(_WORD *)(a1 + 2318);
  v4 = *(_QWORD *)(a1 + 200) == 0;
  *(_BYTE *)(a1 + 524908) = 3;
  *(_WORD *)(a1 + 5890) = 7;
  *(_WORD *)(a1 + 2328) = v57;
  *(_WORD *)(a1 + 2876) = *(_WORD *)(a1 + 2866);
  v15 = *(_QWORD *)a1;
  if ( !v4 )
    goto LABEL_19;
LABEL_144:
  v58 = *(__int64 (**)())(v15 + 176);
  if ( v58 == sub_2FE2F00 )
  {
    v59 = *(_WORD *)(a1 + 2328);
    *(_WORD *)(a1 + 5888) = 12;
    *(_WORD *)(a1 + 2326) = v59;
    *(_WORD *)(a1 + 2874) = *(_WORD *)(a1 + 2876);
    goto LABEL_146;
  }
  if ( !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 (*)(), __int64))v58)(a1, v10, v58, v5) )
  {
    v64 = *(_WORD *)(a1 + 2328);
    *(_WORD *)(a1 + 5888) = 12;
    *(_WORD *)(a1 + 2326) = v64;
    *(_WORD *)(a1 + 2874) = *(_WORD *)(a1 + 2876);
    v15 = *(_QWORD *)a1;
LABEL_146:
    v4 = *(_QWORD *)(a1 + 192) == 0;
    *(_BYTE *)(a1 + 524907) = 8;
    if ( !v4 )
      goto LABEL_20;
LABEL_147:
    v60 = *(_WORD *)(a1 + 2328);
    *(_BYTE *)(a1 + 524906) = 9;
    *(_WORD *)(a1 + 5886) = 12;
    *(_WORD *)(a1 + 2324) = v60;
    *(_WORD *)(a1 + 2872) = *(_WORD *)(a1 + 2876);
    goto LABEL_20;
  }
  v15 = *(_QWORD *)a1;
  v62 = *(__int64 (**)())(*(_QWORD *)a1 + 184LL);
  if ( v62 != sub_2FE2F10 )
  {
    if ( ((unsigned __int8 (__fastcall *)(__int64))v62)(a1) )
    {
      v65 = *(_WORD *)(a1 + 2328);
      *(_WORD *)(a1 + 5888) = 12;
      *(_WORD *)(a1 + 2326) = v65;
      *(_WORD *)(a1 + 2874) = *(_WORD *)(a1 + 2876);
      v15 = *(_QWORD *)a1;
      goto LABEL_153;
    }
    v15 = *(_QWORD *)a1;
  }
  v63 = *(_WORD *)(a1 + 2316);
  *(_WORD *)(a1 + 5888) = 12;
  *(_WORD *)(a1 + 2326) = v63;
  *(_WORD *)(a1 + 2874) = *(_WORD *)(a1 + 2864);
LABEL_153:
  *(_BYTE *)(a1 + 524907) = 9;
LABEL_19:
  if ( !*(_QWORD *)(a1 + 192) )
    goto LABEL_147;
LABEL_20:
  v77 = 18;
  v16 = -159;
  for ( j = 0; j != 212; ++j )
  {
    v18 = (unsigned int)(j + 17);
    if ( !*(_QWORD *)(a1 + 8 * j + 248) )
    {
      v74 = j + 17;
      v19 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(v15 + 168);
      v20 = word_4456580[j + 16];
      v75 = (unsigned __int16)v16 <= 0x34u;
      v78 = word_4456340[j + 16];
      if ( v19 == sub_2FE2EC0 )
      {
        if ( (unsigned __int16)v16 > 0x34u && word_4456340[j + 16] == 1 )
        {
          v72 = 5;
          goto LABEL_90;
        }
        LODWORD(v5) = v78 - 1;
        if ( ((v78 - 1) & v78) != 0 )
        {
          v72 = 7;
LABEL_122:
          _BitScanReverse(&v48, v5);
          v49 = 1 << (32 - (v48 ^ 0x1F));
          if ( (unsigned __int16)v16 > 0x34u )
          {
            v23 = sub_2D43050(v20, v49);
            if ( !v23 )
              goto LABEL_90;
          }
          else
          {
            v23 = sub_2D43AD0(v20, v49);
            if ( !v23 )
              goto LABEL_47;
          }
          LODWORD(v18) = v23;
          goto LABEL_43;
        }
      }
      else
      {
        v44 = v19(a1, v74);
        v18 = (unsigned int)(j + 17);
        v72 = v44;
        if ( (unsigned __int8)v44 > 6u )
        {
          if ( v44 == 7 )
            goto LABEL_42;
          goto LABEL_161;
        }
        if ( (unsigned __int8)v44 > 4u )
          goto LABEL_46;
        if ( v44 != 1 )
LABEL_161:
          BUG();
      }
      v5 = v77;
      if ( (unsigned __int16)((unsigned __int16)v16 < 0x35u ? 207 : 125) < (unsigned __int16)v77 )
        goto LABEL_41;
      v21 = v77;
      do
      {
        v22 = v5;
        if ( (unsigned __int16)(v5 - 17) <= 0xD3u )
          v22 = word_4456580[(unsigned __int16)v5 - 1];
        if ( v22 <= 1u || (unsigned __int16)(v22 - 504) <= 7u || v20 <= 1u || (unsigned __int16)(v20 - 504) <= 7u )
          goto LABEL_161;
        if ( *(_QWORD *)&byte_444C4A0[16 * v20 - 16] < *(_QWORD *)&byte_444C4A0[16 * v22 - 16]
          && (_WORD)v5 != 0
          && (unsigned __int16)(v5 - 176) <= 0x34u == v75
          && word_4456340[(unsigned __int16)v5 - 1] == v78
          && *(_QWORD *)(a1 + 8LL * (unsigned __int16)v5 + 112) )
        {
          *(_WORD *)(a1 + 2 * j + 5900) = v5;
          v18 = 1;
          *(_WORD *)(a1 + 2 * j + 2886) = v5;
          *(_WORD *)(a1 + 2 * j + 2338) = 1;
          *(_BYTE *)(a1 + j + 524913) = 1;
          goto LABEL_45;
        }
        v5 = ++v21;
      }
      while ( (unsigned __int16)((unsigned __int16)v16 < 0x35u ? 207 : 125) >= (unsigned __int16)v21 );
      v18 = (unsigned int)v18;
LABEL_41:
      v72 = 1;
LABEL_42:
      v23 = j + 17;
      if ( v78 )
      {
        v5 = v78 - 1;
        if ( ((unsigned int)v5 & v78) != 0 )
          goto LABEL_122;
        v46 = v77;
        if ( v77 == 229 )
        {
LABEL_127:
          if ( (unsigned __int16)v16 > 0x34u )
            goto LABEL_90;
LABEL_49:
          if ( ((unsigned int)v5 & v78) != 0 )
            goto LABEL_162;
          if ( v78 != 1 )
          {
LABEL_51:
            v24 = v78;
            v68 = j;
            v25 = v20;
            v26 = 1;
            v27 = v78;
            if ( (unsigned __int16)v16 <= 0x34u )
            {
LABEL_52:
              v28 = sub_2D43AD0(v25, v27);
              goto LABEL_53;
            }
            while ( 1 )
            {
              v28 = sub_2D43050(v25, v27);
LABEL_53:
              if ( v28 && *(_QWORD *)(a1 + 8LL * v28 + 112) )
                break;
              v24 >>= 1;
              v26 *= 2;
              if ( v24 == 1 )
                break;
              v27 = v24;
              v25 = v20;
              if ( (unsigned __int16)v16 <= 0x34u )
                goto LABEL_52;
            }
            v76 = v26;
            j = v68;
            goto LABEL_56;
          }
          v76 = 1;
          v24 = 1;
LABEL_57:
          v29 = sub_2D43AD0(v20, v24);
          goto LABEL_58;
        }
        while ( 1 )
        {
          v47 = v46 - 1;
          if ( v20 == word_4456580[v47]
            && v75 == (unsigned __int16)(v46 - 176) <= 0x34u
            && v78 < word_4456340[v47]
            && *(_QWORD *)(a1 + 8LL * v46 + 112) )
          {
            break;
          }
          if ( ++v46 == 229 )
            goto LABEL_127;
        }
        *(_WORD *)(a1 + 2 * j + 5900) = v46;
        *(_WORD *)(a1 + 2 * j + 2886) = v46;
        *(_WORD *)(a1 + 2 * j + 2338) = 1;
        *(_BYTE *)(a1 + j + 524913) = 7;
      }
      else
      {
LABEL_43:
        v18 = (int)v18;
        if ( *(_QWORD *)(a1 + 8LL * (int)v18 + 112) )
        {
          *(_WORD *)(a1 + 2 * j + 5900) = v23;
          v5 = 1;
          *(_BYTE *)(a1 + j + 524913) = 7;
          *(_WORD *)(a1 + 2 * j + 2886) = v23;
          *(_WORD *)(a1 + 2 * j + 2338) = 1;
          goto LABEL_45;
        }
LABEL_46:
        if ( (unsigned __int16)v16 <= 0x34u )
        {
LABEL_47:
          if ( v78 )
          {
            LODWORD(v5) = v78 - 1;
            goto LABEL_49;
          }
LABEL_162:
          BUG();
        }
LABEL_90:
        if ( !v78 )
        {
          v76 = 0;
          goto LABEL_92;
        }
        if ( ((v78 - 1) & v78) == 0 )
        {
          if ( v78 != 1 )
            goto LABEL_51;
          v76 = 1;
          v24 = 1;
LABEL_56:
          if ( (unsigned __int16)v16 > 0x34u )
            goto LABEL_93;
          goto LABEL_57;
        }
        v76 = v78;
LABEL_92:
        v24 = 1;
LABEL_93:
        v29 = sub_2D43050(v20, v24);
LABEL_58:
        if ( v29 && (v30 = v29, *(_QWORD *)(a1 + 8LL * v29 + 112)) )
        {
          if ( (unsigned __int16)(v29 - 17) <= 0xD3u )
            goto LABEL_61;
LABEL_88:
          v31 = v30;
          v32 = v29;
        }
        else
        {
          v29 = v20;
          v30 = v20;
          if ( (unsigned __int16)(v20 - 17) > 0xD3u )
            goto LABEL_88;
LABEL_61:
          v31 = (unsigned __int16)word_4456580[v30 - 1];
          v32 = word_4456580[v30 - 1];
        }
        if ( v32 <= 1u || (unsigned __int16)(v32 - 504) <= 7u )
          goto LABEL_161;
        v33 = 1;
        v34 = *(_QWORD *)&byte_444C4A0[16 * v31 - 16];
        if ( (unsigned int)v34 > 1 )
        {
          _BitScanReverse((unsigned int *)&v34, v34 - 1);
          v33 = 1 << (32 - (v34 ^ 0x1F));
        }
        v80 = 0;
        v5 = *(unsigned __int16 *)(a1 + 2LL * (v30 + 1426));
        v79 = v5;
        if ( (_WORD)v5 != v29 )
        {
          v81 = v29;
          v82 = 0;
          if ( !v29 )
          {
            v70 = v5;
            v69 = v33;
            v35 = sub_3007260(&v81);
            v5 = v70;
            v33 = v69;
            v85 = v35;
            v36 = v35;
            v86 = v37;
            v18 = (unsigned __int8)v37;
            goto LABEL_69;
          }
          if ( v29 == 1 || (unsigned __int16)(v29 - 504) <= 7u )
            goto LABEL_161;
          v45 = &byte_444C4A0[16 * v30 - 16];
          v36 = *(_QWORD *)v45;
          v18 = (unsigned __int8)v45[8];
LABEL_69:
          if ( !(_WORD)v5 )
          {
            v73 = v18;
            v67 = v5;
            v71 = v33;
            v38 = sub_3007260(&v79);
            v18 = v73;
            v5 = v67;
            v83 = v38;
            v33 = v71;
            v84 = v39;
            goto LABEL_71;
          }
          if ( (_WORD)v5 == 1 || (unsigned __int16)(v5 - 504) <= 7u )
            goto LABEL_161;
          v39 = &byte_444C4A0[16 * (unsigned __int16)v5 - 16];
          v38 = *(_QWORD *)v39;
          LOBYTE(v39) = v39[8];
LABEL_71:
          if ( (!(_BYTE)v39 || (_BYTE)v18) && v36 > v38 )
          {
            v40 = v5;
            if ( (unsigned __int16)(v5 - 17) <= 0xD3u )
              v40 = word_4456580[(unsigned __int16)v5 - 1];
            if ( v40 > 1u && (unsigned __int16)(v40 - 504) > 7u )
            {
              v76 *= (unsigned __int16)((unsigned __int64)v33 / *(_QWORD *)&byte_444C4A0[16 * v40 - 16]);
              goto LABEL_79;
            }
            goto LABEL_161;
          }
        }
LABEL_79:
        *(_WORD *)(a1 + 2 * j + 2886) = v5;
        *(_WORD *)(a1 + 2 * j + 2338) = v76;
        if ( ((v78 - 1) & v78) == 0
          || ((_BitScanReverse(&v41, v78 - 1), v42 = 1 << (32 - (v41 ^ 0x1F)), (unsigned __int16)v16 > 0x34u)
            ? (v43 = sub_2D43050(v20, v42))
            : (v43 = sub_2D43AD0(v20, v42)),
              v74 == v43) )
        {
          *(_WORD *)(a1 + 2 * j + 5900) = 1;
          if ( v72 == 5 )
          {
            *(_BYTE *)(a1 + j + 524913) = 5;
          }
          else if ( v72 == 6 || v78 > 1 )
          {
            *(_BYTE *)(a1 + j + 524913) = 6;
          }
          else
          {
            *(_BYTE *)(a1 + j + 524913) = (unsigned __int16)v16 < 0x35u ? 10 : 5;
          }
        }
        else
        {
          *(_WORD *)(a1 + 2 * j + 5900) = v43;
          *(_BYTE *)(a1 + j + 524913) = 7;
        }
      }
LABEL_45:
      v15 = *(_QWORD *)a1;
    }
    ++v77;
    ++v16;
  }
  v50 = 0;
  while ( 1 )
  {
    result = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64, __int64))(v15 + 1272))(
               a1,
               a2,
               (unsigned int)v50,
               v5,
               v18);
    *(_QWORD *)(a1 + 8 * v50 + 3400) = result;
    *(_BYTE *)(a1 + v50++ + 5592) = v52;
    if ( v50 == 274 )
      break;
    v15 = *(_QWORD *)a1;
  }
  return result;
}
