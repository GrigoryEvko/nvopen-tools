// Function: sub_5F6470
// Address: 0x5f6470
//
__int64 __fastcall sub_5F6470(__int64 a1, __int64 *a2)
{
  _QWORD *v4; // r10
  __int64 v5; // rbx
  int v6; // eax
  __int64 v7; // r8
  _QWORD *v8; // r10
  _QWORD *v9; // r11
  __int64 v10; // rcx
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // xmm0_8
  __int64 v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r14
  _QWORD *v20; // rsi
  bool v21; // zf
  __int64 *v22; // rax
  __int64 *v23; // r8
  __int64 *v24; // rax
  __int64 *v25; // rdx
  __int64 v26; // rax
  int v27; // eax
  __int64 v28; // rdi
  __int64 v29; // rdx
  char v30; // al
  __int64 v31; // rdx
  int v32; // eax
  _QWORD *v33; // r10
  __int64 v34; // rax
  int v35; // eax
  __int64 v36; // rdx
  __int64 v37; // rdi
  __int64 v38; // rax
  __int64 v39; // rdx
  int v40; // eax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  int v46; // eax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  _QWORD *v53; // [rsp+0h] [rbp-300h]
  _QWORD *v54; // [rsp+0h] [rbp-300h]
  __int64 v55; // [rsp+8h] [rbp-2F8h]
  _QWORD *v56; // [rsp+8h] [rbp-2F8h]
  __int64 v57; // [rsp+8h] [rbp-2F8h]
  __int64 v58; // [rsp+10h] [rbp-2F0h]
  int v59; // [rsp+10h] [rbp-2F0h]
  __int64 v60; // [rsp+10h] [rbp-2F0h]
  __int64 v61; // [rsp+10h] [rbp-2F0h]
  _BOOL4 v62; // [rsp+18h] [rbp-2E8h]
  _QWORD *v63; // [rsp+18h] [rbp-2E8h]
  int v64; // [rsp+18h] [rbp-2E8h]
  __int64 v65; // [rsp+18h] [rbp-2E8h]
  _QWORD *v66; // [rsp+18h] [rbp-2E8h]
  _QWORD *v67; // [rsp+18h] [rbp-2E8h]
  _QWORD *v68; // [rsp+18h] [rbp-2E8h]
  unsigned int v69; // [rsp+24h] [rbp-2DCh]
  __int64 v70; // [rsp+28h] [rbp-2D8h]
  __int64 v71; // [rsp+30h] [rbp-2D0h] BYREF
  int v72; // [rsp+38h] [rbp-2C8h] BYREF
  unsigned __int64 v73; // [rsp+40h] [rbp-2C0h] BYREF
  __int64 v74; // [rsp+48h] [rbp-2B8h]
  __m128i v75; // [rsp+50h] [rbp-2B0h]
  __m128i v76; // [rsp+60h] [rbp-2A0h]
  __m128i v77; // [rsp+70h] [rbp-290h]
  _QWORD v78[70]; // [rsp+80h] [rbp-280h] BYREF
  __int16 v79; // [rsp+2B0h] [rbp-50h]

  v69 = dword_4F04C3C;
  if ( a2 )
  {
    v4 = (__int64 *)((char *)a2 + 36);
    v62 = (a2[4] & 8) != 0;
  }
  else
  {
    v62 = 0;
    v4 = (_QWORD *)(a1 + 44);
  }
  v5 = 0;
  dword_4F04C3C = 1;
  v6 = dword_4F04C64;
  if ( dword_4F04C64 != -1 )
    goto LABEL_6;
  while ( (unsigned __int8)(*(_BYTE *)(v5 + 4) - 6) > 1u || *(_QWORD *)(v5 + 208) != *(_QWORD *)(a1 + 8) )
  {
    v6 = *(_DWORD *)(v5 + 552);
    if ( v6 == -1 )
      BUG();
LABEL_6:
    v5 = qword_4F04C68[0] + 776LL * v6;
  }
  v70 = *(_QWORD *)(v5 + 600);
  sub_5E4C60((__int64)v78, v4);
  v10 = *v8;
  v11 = 0xA3A0FD5C5F02A3A1LL * ((v5 - *v9) >> 3);
  v12 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
  v73 = v12;
  v75 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v74 = v10;
  v76 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v77 = _mm_loadu_si128(&xmmword_4F06660[3]);
  if ( !a2 )
  {
    v15 = sub_72CBE0();
    v16 = sub_72D2E0(v15, 0);
    LOBYTE(v79) = v79 | 0x20;
    v78[36] = v16;
    v71 = v16;
    *(_BYTE *)(v70 + 12) = 2;
    v17 = sub_5F4F20((__int64)&v73, v70, (__int64)v78, v11);
    *(_BYTE *)(v70 + 12) = 0;
    v18 = v17;
    dword_4F04C3C = v69;
    return v18;
  }
  if ( (a2[4] & 1) != 0 )
  {
    sub_878710(*(_QWORD *)a2[2], &v73);
    v20 = (_QWORD *)a2[2];
    v21 = (a2[4] & 0x60) == 0;
    v71 = v20[36];
    qmemcpy(v78, v20, 0x1D8u);
    v78[19] = v78;
    HIBYTE(v79) = (16 * !v21) | HIBYTE(v79) & 0xEF;
    goto LABEL_23;
  }
  if ( v7 )
  {
    HIBYTE(v79) = (*(_BYTE *)(v7 + 175) >> 2) & 0x10 | HIBYTE(v79) & 0xEF;
    if ( (*(_BYTE *)(v7 + 172) & 1) == 0 )
    {
      if ( *(_QWORD *)v7 )
      {
        v73 = **(_QWORD **)v7;
      }
      else
      {
        v73 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
        v75 = _mm_loadu_si128(&xmmword_4F06660[1]);
        v76 = _mm_loadu_si128(&xmmword_4F06660[2]);
        v75.m128i_i8[1] |= 0x20u;
        v74 = *(_QWORD *)dword_4F07508;
        v77 = _mm_loadu_si128(&xmmword_4F06660[3]);
      }
      v13 = *(_QWORD *)(v7 + 120);
      goto LABEL_16;
    }
    if ( v62 )
    {
      v26 = a2[2];
      if ( v26 )
      {
        v63 = v8;
        v71 = *(_QWORD *)(v26 + 120);
        v27 = sub_8D2E30(v71);
        v8 = v63;
        if ( v27 )
        {
          v28 = v71;
        }
        else
        {
          v45 = sub_72D2E0(v71, 0);
          v8 = v63;
          v71 = v45;
          v28 = v45;
        }
      }
      else
      {
        v28 = *(_QWORD *)(v7 + 120);
        v71 = v28;
      }
    }
    else
    {
      v68 = v8;
      v42 = sub_8D46C0(*(_QWORD *)(v7 + 120));
      v21 = (*(_BYTE *)(a1 + 24) & 2) == 0;
      v8 = v68;
      v71 = v42;
      v28 = v42;
      if ( v21 )
      {
        v43 = sub_73C570(v42, 1, -1);
        v8 = v68;
        v71 = v43;
        v28 = v43;
      }
      else
      {
        while ( *(_BYTE *)(v28 + 140) == 12 )
          v28 = *(_QWORD *)(v28 + 160);
        v71 = v28;
      }
    }
    v66 = v8;
    v40 = sub_8D32E0(v28);
    v33 = v66;
    if ( v40 )
    {
      v44 = sub_8D46C0(v71);
      v33 = v66;
      v71 = v44;
      v78[36] = v44;
    }
    else
    {
      v78[36] = v71;
    }
LABEL_67:
    v79 |= 0x820u;
    *(_BYTE *)(v70 + 12) = 2;
    v67 = v33;
    v18 = sub_5F4F20((__int64)&v73, v70, (__int64)v78, v11);
    v41 = sub_87F7E0(8, v67);
    v72 = 0;
    *(_QWORD *)(v41 + 88) = v18;
    sub_885620(v41, (unsigned int)v11, &v72);
    goto LABEL_27;
  }
  v29 = a2[2];
  if ( v29 )
  {
    v30 = *(_BYTE *)(v29 + 145);
    v59 = 1;
    if ( (v30 & 1) == 0 )
    {
      v54 = v8;
      v57 = a2[2];
      sub_878710(*(_QWORD *)v29, &v73);
      v29 = v57;
      v59 = 0;
      v8 = v54;
      v30 = *(_BYTE *)(v57 + 145);
    }
    HIBYTE(v79) = HIBYTE(v79) & 0xEF | (8 * v30) & 0x10;
    v31 = *(_QWORD *)(v29 + 120);
    v71 = v31;
    goto LABEL_47;
  }
  v56 = v8;
  if ( (a2[4] & 4) != 0 )
  {
    v46 = sub_830940(&v72, &v71, 0, a2[4] & 4);
    v8 = v56;
    v59 = v46;
    if ( v62 )
    {
      v31 = v71;
    }
    else
    {
      v52 = sub_8D46C0(v71);
      v8 = v56;
      v71 = v52;
      v31 = v52;
    }
LABEL_47:
    v53 = v8;
    v55 = v31;
    v32 = sub_8D32E0(v31);
    v33 = v53;
    if ( v32 )
    {
      v47 = sub_8D46C0(v71);
      v39 = v55;
      v71 = v47;
      v33 = v53;
      v14 = v47;
      if ( !v59 )
        goto LABEL_61;
    }
    else
    {
      if ( !v59 )
      {
LABEL_17:
        v14 = v71;
        if ( !v62 )
        {
          v78[36] = v71;
          goto LABEL_23;
        }
LABEL_54:
        v60 = a2[2];
        v35 = sub_8D4070(v14);
        v36 = v60;
        v64 = v35;
        if ( v35 )
        {
          v48 = sub_8D4050(v71);
          v36 = v60;
          v71 = v48;
          v37 = v48;
          if ( !v60 )
          {
LABEL_59:
            v71 = sub_72D2E0(v37, 0);
            v78[36] = v71;
            goto LABEL_23;
          }
          v64 = 1;
        }
        else if ( !v60 )
        {
          v37 = v71;
LABEL_77:
          v71 = sub_72D600(v37);
          v78[36] = v71;
          goto LABEL_23;
        }
        v61 = v36;
        if ( (unsigned int)sub_8D2FB0(*(_QWORD *)(v36 + 120)) )
        {
          v71 = sub_8D46C0(*(_QWORD *)(v61 + 120));
          v37 = v71;
        }
        else
        {
          v49 = sub_72F130(*(_QWORD *)(*(_QWORD *)(v61 + 40) + 32LL));
          v37 = v71;
          v50 = *(_QWORD *)(v49 + 152);
          if ( *(_BYTE *)(v50 + 140) == 7 && (*(_BYTE *)(*(_QWORD *)(v50 + 168) + 18LL) & 0x7F) == 1 )
          {
            v71 = sub_73C570(v71, 1, -1);
            v37 = v71;
          }
        }
        if ( v64 )
          goto LABEL_59;
        goto LABEL_77;
      }
      v14 = v71;
    }
    v78[36] = v14;
    goto LABEL_67;
  }
  v73 = v12;
  v75.m128i_i8[1] |= 0x20u;
  v74 = *(_QWORD *)dword_4F07508;
  v13 = sub_72C930();
LABEL_16:
  v71 = v13;
  v58 = v13;
  if ( !(unsigned int)sub_8D32E0(v13) )
    goto LABEL_17;
  v38 = sub_8D46C0(v71);
  v39 = v58;
  v71 = v38;
  v14 = v38;
LABEL_61:
  if ( v62 )
    goto LABEL_54;
  v65 = v39;
  if ( (unsigned int)sub_8D2310(v14) )
  {
    v71 = v65;
    v78[36] = v65;
  }
  else
  {
    v78[36] = v71;
  }
LABEL_23:
  *(_BYTE *)(v70 + 12) = 2;
  v18 = sub_5F4F20((__int64)&v73, v70, (__int64)v78, v11);
  if ( (a2[4] & 1) != 0 )
  {
    if ( *(_BYTE *)(v78[0] + 80LL) == 8 )
      *(_BYTE *)(*(_QWORD *)(v78[0] + 88LL) + 144LL) |= 0x80u;
    sub_63B800(a2, v78);
    if ( (*(_BYTE *)(a1 + 25) & 8) != 0 )
    {
      v34 = a2[1];
      if ( *(_BYTE *)(v34 + 48) == 3 )
      {
        v51 = *(_QWORD *)(v34 + 56);
        if ( *(_BYTE *)(v51 + 24) == 1 && *(_BYTE *)(v51 + 56) == 21 )
          sub_6849F0(7, 3596, (char *)a2 + 36, "__device__");
      }
      sub_5E5D30(v71, (__int64 *)((char *)a2 + 36), 1);
      if ( (unsigned int)sub_8D3C40(v71) )
        sub_6849F0(7, 3616, (char *)a2 + 36, "__device__");
    }
  }
  else
  {
    *(_BYTE *)(v78[0] + 83LL) |= 0x40u;
  }
LABEL_27:
  *(_BYTE *)(v70 + 12) = 0;
  dword_4F04C3C = v69;
  if ( (a2[4] & 0x10) == 0 || !*a2 )
    return v18;
  v22 = *(__int64 **)a1;
  if ( *(__int64 **)a1 == a2 )
  {
    v23 = (__int64 *)a1;
    v24 = (__int64 *)*a2;
  }
  else
  {
    do
    {
      v23 = v22;
      v22 = (__int64 *)*v22;
    }
    while ( a2 != v22 );
    v24 = (__int64 *)*a2;
    if ( !*a2 )
      return v18;
  }
  v25 = 0;
  do
  {
    if ( (*((_BYTE *)v24 + 33) & 2) == 0 )
      v25 = v24;
    v24 = (__int64 *)*v24;
  }
  while ( v24 );
  if ( v25 )
  {
    *v23 = *a2;
    *a2 = *v25;
    *v25 = (__int64)a2;
  }
  return v18;
}
