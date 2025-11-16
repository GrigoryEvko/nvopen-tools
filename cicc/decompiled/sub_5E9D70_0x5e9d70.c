// Function: sub_5E9D70
// Address: 0x5e9d70
//
__int64 __fastcall sub_5E9D70(__int64 a1, __int64 a2, __int64 a3, int a4, _QWORD *a5)
{
  __int64 v5; // r13
  int v6; // r15d
  __int64 v7; // rax
  __int64 v8; // r11
  __int64 v9; // r10
  bool v10; // r14
  __int64 v11; // rbx
  __int64 v12; // r15
  char v13; // al
  char v14; // al
  __int64 v15; // r12
  char v16; // al
  __int64 v18; // rsi
  __int64 v19; // r10
  __int64 i; // rax
  __int64 v21; // r11
  int v22; // eax
  __int64 v23; // rax
  __int64 v24; // rax
  int v25; // eax
  __m128i *v26; // r10
  __int64 v27; // rsi
  int v28; // eax
  __int64 v29; // rax
  char v30; // al
  char v31; // al
  __int64 v32; // rax
  int v33; // eax
  __int64 v34; // rdx
  __int64 v35; // rax
  int v36; // eax
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r11
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rdi
  __int64 v49; // r11
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // [rsp-10h] [rbp-170h]
  __int64 v53; // [rsp+0h] [rbp-160h]
  __int64 v54; // [rsp+10h] [rbp-150h]
  __int64 v55; // [rsp+10h] [rbp-150h]
  __int64 v56; // [rsp+10h] [rbp-150h]
  __int64 v57; // [rsp+10h] [rbp-150h]
  const __m128i *v58; // [rsp+10h] [rbp-150h]
  __int64 v59; // [rsp+10h] [rbp-150h]
  __int64 v60; // [rsp+10h] [rbp-150h]
  __int64 v61; // [rsp+18h] [rbp-148h]
  __int64 v63; // [rsp+28h] [rbp-138h]
  __int64 v64; // [rsp+28h] [rbp-138h]
  __int64 v65; // [rsp+30h] [rbp-130h]
  __int64 v66; // [rsp+30h] [rbp-130h]
  int v67; // [rsp+38h] [rbp-128h]
  __int64 v68; // [rsp+38h] [rbp-128h]
  __int64 v70; // [rsp+48h] [rbp-118h]
  __int64 v71; // [rsp+50h] [rbp-110h]
  char v72; // [rsp+5Eh] [rbp-102h]
  __int64 v73; // [rsp+60h] [rbp-100h]
  int v75; // [rsp+70h] [rbp-F0h]
  __int64 v76; // [rsp+70h] [rbp-F0h]
  int v77; // [rsp+78h] [rbp-E8h]
  __int64 v78; // [rsp+78h] [rbp-E8h]
  int v79; // [rsp+8Ch] [rbp-D4h] BYREF
  __int64 v80; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v81; // [rsp+98h] [rbp-C8h] BYREF
  __m128i v82; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v83; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v84; // [rsp+B8h] [rbp-A8h]
  __int64 v85; // [rsp+C0h] [rbp-A0h]
  _BYTE v86[144]; // [rsp+D0h] [rbp-90h] BYREF

  v5 = a1;
  v70 = *(_QWORD *)(a2 + 288);
  v61 = *(_QWORD *)(a1 + 64);
  if ( a5 )
    *a5 = 0;
  v6 = 0;
  if ( *(_BYTE *)(a1 + 80) == 17 )
  {
    v5 = *(_QWORD *)(a1 + 88);
    v6 = 1;
  }
  v7 = v70;
  if ( *(_BYTE *)(v70 + 140) == 12 )
  {
    do
      v7 = *(_QWORD *)(v7 + 160);
    while ( *(_BYTE *)(v7 + 140) == 12 );
  }
  else
  {
    v7 = v70;
  }
  v8 = *(_QWORD *)(v7 + 168);
  v9 = *(_QWORD *)(v8 + 40);
  v72 = *(_BYTE *)(v8 + 18) & 0x7F;
  if ( !unk_4D0487C
    || (*(_BYTE *)(v8 + 18) & 1) != 0
    || (*(_BYTE *)(a2 + 10) & 8) == 0
    || (v76 = *(_QWORD *)(v8 + 40), v78 = *(_QWORD *)(v7 + 168), v30 = sub_877F80(v5), v8 = v78, v9 = v76, v30 == 1) )
  {
    if ( !v5 )
      return 0;
    v10 = 0;
  }
  else
  {
    v31 = sub_877F80(v5);
    v8 = v78;
    v9 = v76;
    v10 = v31 != 2;
    if ( !v5 )
    {
LABEL_19:
      if ( v10 )
      {
        v16 = *(_BYTE *)(v8 + 18);
        *(_QWORD *)(v8 + 40) = v9;
        *(_BYTE *)(v8 + 18) = v72 | v16 & 0x80;
      }
      return 0;
    }
  }
  v77 = v6;
  v11 = v8;
  v12 = v9;
  v75 = v72 != 0;
  while ( 1 )
  {
    if ( v10 )
    {
      v13 = *(_BYTE *)(v11 + 18);
      *(_QWORD *)(v11 + 40) = v12;
      *(_BYTE *)(v11 + 18) = v72 & 0x7F | v13 & 0x80;
      *(_BYTE *)(v11 + 21) = (v12 != 0) | *(_BYTE *)(v11 + 21) & 0xFE;
      v75 = v72 != 0;
    }
    v14 = *(_BYTE *)(v5 + 80);
    v15 = v5;
    if ( v14 == 16 )
    {
      if ( (*(_BYTE *)(v5 + 96) & 4) == 0 )
        goto LABEL_16;
      v15 = **(_QWORD **)(v5 + 88);
      v14 = *(_BYTE *)(v15 + 80);
      if ( v14 == 24 )
      {
        v15 = *(_QWORD *)(v15 + 88);
        v14 = *(_BYTE *)(v15 + 80);
      }
      if ( v14 != 20 )
        break;
    }
    if ( (v14 == 20) == a4 )
      goto LABEL_28;
LABEL_16:
    if ( v77 )
    {
      v5 = *(_QWORD *)(v5 + 8);
      if ( v5 )
        continue;
    }
    v8 = v11;
    v9 = v12;
    goto LABEL_19;
  }
  if ( v14 != 10 || a4 )
    goto LABEL_16;
LABEL_28:
  v18 = *(_QWORD *)(v15 + 88);
  v73 = v18;
  if ( v14 == 20 )
    v73 = *(_QWORD *)(v18 + 176);
  v19 = *(_QWORD *)(v73 + 152);
  for ( i = v19; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v21 = *(_QWORD *)(i + 168);
  v71 = *(_QWORD *)(v21 + 40);
  v22 = (*(_BYTE *)(v21 + 18) & 0x7F) != 0;
  if ( v71 && v10 )
  {
    if ( !v12 )
    {
      v18 = v61;
      *(_BYTE *)(v11 + 21) |= 1u;
      *(_QWORD *)(v11 + 40) = v61;
    }
    *(_BYTE *)(v11 + 18) |= 1u;
    v75 = 1;
  }
  if ( v75 != v22 || ((*(_BYTE *)(v21 + 19) ^ *(_BYTE *)(v11 + 19)) & 0xC0) != 0 )
    goto LABEL_16;
  if ( *(_BYTE *)(v15 + 80) == 20 )
  {
    if ( !a3 )
      goto LABEL_16;
    v34 = *(_QWORD *)(v15 + 88);
    v66 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v34 + 104) + 176LL) + 16LL);
    v35 = *(_QWORD *)(*(_QWORD *)(unk_4F04C68 + 776LL * dword_4F04C64 + 616) + 488LL);
    v68 = v35;
    if ( v35 )
      v68 = *(_QWORD *)(v35 + 16);
    v56 = v21;
    v64 = v19;
    if ( !(unsigned int)sub_89B3C0(**(_QWORD **)(v34 + 328), a3, 0, 0, 0, 8) )
      goto LABEL_16;
    v18 = v68;
    v36 = sub_739400(v66, v68);
    v19 = v64;
    v21 = v56;
    if ( !v36 )
      goto LABEL_16;
  }
  if ( !v75 || (v67 = 0, (*(_BYTE *)(a2 + 131) & 2) != 0) )
  {
    *(_QWORD *)(v11 + 40) = 0;
    v67 = 1;
    *(_QWORD *)(v21 + 40) = 0;
  }
  v23 = *(_QWORD *)(v21 + 56);
  *(_QWORD *)(v21 + 56) = 0;
  v65 = v23;
  v24 = *(_QWORD *)(v11 + 56);
  *(_QWORD *)(v11 + 56) = 0;
  v63 = v24;
  if ( v19 != v70 )
  {
    v18 = v70;
    v54 = v21;
    v25 = sub_8DED30(v19, v70, 3411976);
    v21 = v54;
    if ( !v25 )
    {
LABEL_56:
      *(_QWORD *)(v21 + 56) = v65;
      *(_QWORD *)(v11 + 56) = v63;
      if ( v67 )
      {
        *(_QWORD *)(v11 + 40) = v12;
        *(_QWORD *)(v21 + 40) = v71;
      }
      goto LABEL_16;
    }
  }
  v26 = *(__m128i **)(v73 + 216);
  if ( v26 )
  {
    if ( (*(_BYTE *)(a2 + 130) & 0xC) == 0 )
    {
      if ( *(_BYTE *)(v15 + 80) == 20 && (*(_BYTE *)(v61 + 177) & 0x30) == 0x10 )
      {
        v53 = v21;
        v58 = *(const __m128i **)(v73 + 216);
        v79 = 0;
        v83 = 0;
        v84 = 0;
        v85 = 0;
        v37 = sub_823970(0);
        v84 = 0;
        v83 = v37;
        v80 = 0;
        v81 = sub_724DC0(0, v18, v38, v39, v40, v41);
        v82 = _mm_loadu_si128(v58);
        sub_892150(v86);
        sub_89F970(v61, &v83);
        v42 = sub_6EFFF0(
                v58->m128i_i64[0],
                (unsigned int)&v83,
                (unsigned int)v86,
                0x4000,
                v81,
                (unsigned int)&v80,
                (__int64)&v79);
        v82.m128i_i64[0] = v42;
        if ( v79 || v42 )
        {
          sub_724E30(&v81);
          v45 = v53;
          if ( v79 )
          {
            v46 = sub_72C390();
            v47 = sub_73A7B0(v46);
            v45 = v53;
            v82.m128i_i64[0] = v47;
          }
        }
        else
        {
          if ( v80 )
          {
            sub_724E30(&v81);
            v48 = v80;
            v49 = v53;
          }
          else
          {
            v51 = ((__int64 (__fastcall *)(__int64 *, __int64 *, __int64, __int64, __int64))sub_724E50)(
                    &v81,
                    &v83,
                    v43,
                    v44,
                    v52);
            v49 = v53;
            v80 = v51;
            v48 = v51;
          }
          v60 = v49;
          v50 = sub_73A720(v48);
          v45 = v60;
          v82.m128i_i64[0] = v50;
        }
        v59 = v45;
        sub_823A00(v83, 24 * v84);
        v21 = v59;
        v26 = &v82;
        v27 = *(_QWORD *)(a2 + 400);
      }
      else
      {
        v27 = *(_QWORD *)(a2 + 400);
      }
      goto LABEL_48;
    }
LABEL_64:
    if ( (*(_BYTE *)(v15 + 104) & 1) != 0 )
    {
      v57 = v21;
      v33 = sub_8796F0(v15);
      v21 = v57;
    }
    else
    {
      v32 = *(_QWORD *)(v15 + 88);
      if ( *(_BYTE *)(v15 + 80) == 20 )
        v32 = *(_QWORD *)(v32 + 176);
      v33 = (*(_BYTE *)(v32 + 208) & 4) != 0;
    }
    if ( v33 )
      goto LABEL_56;
LABEL_69:
    *(_QWORD *)(v21 + 56) = v65;
    *(_QWORD *)(v11 + 56) = v63;
    if ( v67 )
    {
      *(_QWORD *)(v11 + 40) = v12;
      *(_QWORD *)(v21 + 40) = v71;
    }
  }
  else
  {
    v27 = *(_QWORD *)(a2 + 400);
    if ( !v27 )
      goto LABEL_69;
    if ( (*(_BYTE *)(a2 + 130) & 0xC) != 0 )
      goto LABEL_64;
LABEL_48:
    v55 = v21;
    v28 = sub_739400(v26, v27);
    *(_QWORD *)(v55 + 56) = v65;
    *(_QWORD *)(v11 + 56) = v63;
    if ( v67 )
    {
      *(_QWORD *)(v11 + 40) = v12;
      *(_QWORD *)(v55 + 40) = v71;
    }
    if ( !v28 )
      goto LABEL_16;
  }
  v29 = 0;
  if ( (*(_BYTE *)(v73 + 194) & 0x40) != 0 )
    v29 = *(_QWORD *)(v73 + 232);
  if ( *(_QWORD *)(a2 + 320) != v29 )
    goto LABEL_16;
  return v5;
}
