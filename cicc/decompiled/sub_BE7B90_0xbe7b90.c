// Function: sub_BE7B90
// Address: 0xbe7b90
//
void __fastcall sub_BE7B90(__int64 *a1, __int64 a2, _BYTE *a3)
{
  __int64 *v3; // rbx
  __int64 v4; // rax
  __int64 *v5; // r14
  bool v6; // r13
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // rax
  __int64 v44; // rdx
  __m128i *v45; // rax
  __int64 v46; // rcx
  __m128i *v47; // rax
  __int64 v48; // rcx
  __int64 v49; // rax
  __int64 v50; // rdx
  _BYTE *v51; // [rsp+0h] [rbp-E0h] BYREF
  __int64 v52[2]; // [rsp+8h] [rbp-D8h] BYREF
  __int64 v53; // [rsp+18h] [rbp-C8h] BYREF
  _QWORD v54[2]; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v55; // [rsp+30h] [rbp-B0h] BYREF
  __m128i *v56; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v57; // [rsp+48h] [rbp-98h]
  __m128i v58; // [rsp+50h] [rbp-90h] BYREF
  _QWORD v59[2]; // [rsp+60h] [rbp-80h] BYREF
  __m128i v60; // [rsp+70h] [rbp-70h] BYREF
  _QWORD v61[2]; // [rsp+80h] [rbp-60h] BYREF
  __int64 v62; // [rsp+90h] [rbp-50h]
  __int64 v63; // [rsp+98h] [rbp-48h]
  __int16 v64; // [rsp+A0h] [rbp-40h]

  v52[0] = a2;
  v51 = a3;
  v3 = (__int64 *)sub_A73280(v52);
  v4 = sub_A73290(v52);
  if ( v3 == (__int64 *)v4 )
    return;
  v5 = (__int64 *)v4;
  while ( 1 )
  {
    v53 = *v3;
    if ( !sub_A71840((__int64)&v53) )
      break;
    v7 = sub_A71FD0(&v53);
    if ( v8 == 19
      && !(*(_QWORD *)v7 ^ 0x662D786F72707061LL | *(_QWORD *)(v7 + 8) ^ 0x6D2D70662D636E75LL)
      && *(_WORD *)(v7 + 16) == 29793
      && *(_BYTE *)(v7 + 18) == 104 )
    {
      v49 = sub_A72240(&v53);
      if ( v50 )
      {
        if ( v50 == 4 )
        {
          if ( *(_DWORD *)v49 == 1702195828 )
            goto LABEL_9;
LABEL_109:
          v62 = v49;
          v61[0] = "invalid value for 'approx-func-fp-math' attribute: ";
          v63 = v50;
          v64 = 1283;
          sub_BDBF70(a1, (__int64)v61);
          goto LABEL_9;
        }
        if ( v50 != 5 || *(_DWORD *)v49 != 1936482662 || *(_BYTE *)(v49 + 4) != 101 )
          goto LABEL_109;
      }
    }
LABEL_9:
    v9 = sub_A71FD0(&v53);
    if ( v10 == 18
      && !(*(_QWORD *)v9 ^ 0x6572702D7373656CLL | *(_QWORD *)(v9 + 8) ^ 0x6D70662D65736963LL)
      && *(_WORD *)(v9 + 16) == 25697 )
    {
      v43 = sub_A72240(&v53);
      if ( v44 )
      {
        if ( v44 == 4 )
        {
          if ( *(_DWORD *)v43 == 1702195828 )
            goto LABEL_10;
LABEL_91:
          v62 = v43;
          v61[0] = "invalid value for 'less-precise-fpmad' attribute: ";
          v63 = v44;
          v64 = 1283;
          sub_BDBF70(a1, (__int64)v61);
          goto LABEL_10;
        }
        if ( v44 != 5 || *(_DWORD *)v43 != 1936482662 || *(_BYTE *)(v43 + 4) != 101 )
          goto LABEL_91;
      }
    }
LABEL_10:
    v11 = sub_A71FD0(&v53);
    if ( v12 == 15
      && *(_QWORD *)v11 == 0x2D73666E692D6F6ELL
      && *(_DWORD *)(v11 + 8) == 1831694438
      && *(_WORD *)(v11 + 12) == 29793
      && *(_BYTE *)(v11 + 14) == 104 )
    {
      v42 = sub_A72240(&v53);
      if ( v41 )
      {
        if ( v41 == 4 )
        {
          if ( *(_DWORD *)v42 == 1702195828 )
            goto LABEL_11;
LABEL_84:
          v62 = v42;
          v61[0] = "invalid value for 'no-infs-fp-math' attribute: ";
          v63 = v41;
          v64 = 1283;
          sub_BDBF70(a1, (__int64)v61);
          goto LABEL_11;
        }
        if ( v41 != 5 || *(_DWORD *)v42 != 1936482662 || *(_BYTE *)(v42 + 4) != 101 )
          goto LABEL_84;
      }
    }
LABEL_11:
    v13 = sub_A71FD0(&v53);
    if ( v14 == 21
      && !(*(_QWORD *)v13 ^ 0x6E696C6E692D6F6ELL | *(_QWORD *)(v13 + 8) ^ 0x742D656E696C2D65LL)
      && *(_DWORD *)(v13 + 16) == 1701601889
      && *(_BYTE *)(v13 + 20) == 115 )
    {
      v39 = sub_A72240(&v53);
      if ( v40 )
      {
        if ( v40 == 4 )
        {
          if ( *(_DWORD *)v39 == 1702195828 )
            goto LABEL_12;
LABEL_75:
          v62 = v39;
          v61[0] = "invalid value for 'no-inline-line-tables' attribute: ";
          v63 = v40;
          v64 = 1283;
          sub_BDBF70(a1, (__int64)v61);
          goto LABEL_12;
        }
        if ( v40 != 5 || *(_DWORD *)v39 != 1936482662 || *(_BYTE *)(v39 + 4) != 101 )
          goto LABEL_75;
      }
    }
LABEL_12:
    v15 = sub_A71FD0(&v53);
    if ( v16 == 14
      && *(_QWORD *)v15 == 0x2D706D756A2D6F6ELL
      && *(_DWORD *)(v15 + 8) == 1818386804
      && *(_WORD *)(v15 + 12) == 29541 )
    {
      v37 = sub_A72240(&v53);
      if ( v38 )
      {
        if ( v38 == 4 )
        {
          if ( *(_DWORD *)v37 == 1702195828 )
            goto LABEL_13;
LABEL_67:
          v62 = v37;
          v61[0] = "invalid value for 'no-jump-tables' attribute: ";
          v63 = v38;
          v64 = 1283;
          sub_BDBF70(a1, (__int64)v61);
          goto LABEL_13;
        }
        if ( v38 != 5 || *(_DWORD *)v37 != 1936482662 || *(_BYTE *)(v37 + 4) != 101 )
          goto LABEL_67;
      }
    }
LABEL_13:
    v17 = sub_A71FD0(&v53);
    if ( v18 == 15
      && *(_QWORD *)v17 == 0x2D736E616E2D6F6ELL
      && *(_DWORD *)(v17 + 8) == 1831694438
      && *(_WORD *)(v17 + 12) == 29793
      && *(_BYTE *)(v17 + 14) == 104 )
    {
      v36 = sub_A72240(&v53);
      if ( v35 )
      {
        if ( v35 == 4 )
        {
          if ( *(_DWORD *)v36 == 1702195828 )
            goto LABEL_14;
LABEL_59:
          v62 = v36;
          v61[0] = "invalid value for 'no-nans-fp-math' attribute: ";
          v63 = v35;
          v64 = 1283;
          sub_BDBF70(a1, (__int64)v61);
          goto LABEL_14;
        }
        if ( v35 != 5 || *(_DWORD *)v36 != 1936482662 || *(_BYTE *)(v36 + 4) != 101 )
          goto LABEL_59;
      }
    }
LABEL_14:
    v19 = sub_A71FD0(&v53);
    if ( v20 == 23
      && !(*(_QWORD *)v19 ^ 0x656E6769732D6F6ELL | *(_QWORD *)(v19 + 8) ^ 0x2D736F72657A2D64LL)
      && *(_DWORD *)(v19 + 16) == 1831694438
      && *(_WORD *)(v19 + 20) == 29793
      && *(_BYTE *)(v19 + 22) == 104 )
    {
      v33 = sub_A72240(&v53);
      if ( v34 )
      {
        if ( v34 == 4 )
        {
          if ( *(_DWORD *)v33 == 1702195828 )
            goto LABEL_15;
LABEL_50:
          v62 = v33;
          v61[0] = "invalid value for 'no-signed-zeros-fp-math' attribute: ";
          v64 = 1283;
          v63 = v34;
          sub_BDBF70(a1, (__int64)v61);
          goto LABEL_15;
        }
        if ( v34 != 5 || *(_DWORD *)v33 != 1936482662 || *(_BYTE *)(v33 + 4) != 101 )
          goto LABEL_50;
      }
    }
LABEL_15:
    v21 = sub_A71FD0(&v53);
    if ( v22 == 23
      && !(*(_QWORD *)v21 ^ 0x2D656C69666F7270LL | *(_QWORD *)(v21 + 8) ^ 0x612D656C706D6173LL)
      && *(_DWORD *)(v21 + 16) == 1920295779
      && *(_WORD *)(v21 + 20) == 29793
      && *(_BYTE *)(v21 + 22) == 101 )
    {
      v31 = sub_A72240(&v53);
      if ( v32 )
      {
        if ( v32 == 4 )
        {
          if ( *(_DWORD *)v31 == 1702195828 )
            goto LABEL_16;
LABEL_41:
          v62 = v31;
          v61[0] = "invalid value for 'profile-sample-accurate' attribute: ";
          v64 = 1283;
          v63 = v32;
          sub_BDBF70(a1, (__int64)v61);
          goto LABEL_16;
        }
        if ( v32 != 5 || *(_DWORD *)v31 != 1936482662 || *(_BYTE *)(v31 + 4) != 101 )
          goto LABEL_41;
      }
    }
LABEL_16:
    v23 = sub_A71FD0(&v53);
    if ( v24 != 14 )
      goto LABEL_17;
    if ( *(_QWORD *)v23 != 0x662D656661736E75LL )
      goto LABEL_17;
    if ( *(_DWORD *)(v23 + 8) != 1634545008 )
      goto LABEL_17;
    if ( *(_WORD *)(v23 + 12) != 26740 )
      goto LABEL_17;
    v29 = sub_A72240(&v53);
    if ( !v30 )
      goto LABEL_17;
    if ( v30 != 4 )
    {
      if ( v30 == 5 && *(_DWORD *)v29 == 1936482662 && *(_BYTE *)(v29 + 4) == 101 )
        goto LABEL_17;
LABEL_32:
      v62 = v29;
      v61[0] = "invalid value for 'unsafe-fp-math' attribute: ";
      v63 = v30;
      v64 = 1283;
      sub_BDBF70(a1, (__int64)v61);
      goto LABEL_17;
    }
    if ( *(_DWORD *)v29 != 1702195828 )
      goto LABEL_32;
LABEL_17:
    v25 = sub_A71FD0(&v53);
    if ( v26 != 18 )
      goto LABEL_4;
    if ( *(_QWORD *)v25 ^ 0x706D61732D657375LL | *(_QWORD *)(v25 + 8) ^ 0x69666F72702D656CLL )
      goto LABEL_4;
    if ( *(_WORD *)(v25 + 16) != 25964 )
      goto LABEL_4;
    v27 = sub_A72240(&v53);
    if ( !v28 )
      goto LABEL_4;
    if ( v28 == 4 )
    {
      if ( *(_DWORD *)v27 == 1702195828 )
        goto LABEL_4;
    }
    else if ( v28 == 5 && *(_DWORD *)v27 == 1936482662 && *(_BYTE *)(v27 + 4) == 101 )
    {
      goto LABEL_4;
    }
    v63 = v28;
    v61[0] = "invalid value for 'use-sample-profile' attribute: ";
    v62 = v27;
    v64 = 1283;
    sub_BDBF70(a1, (__int64)v61);
LABEL_4:
    if ( v5 == ++v3 )
      return;
  }
  v6 = sub_A71820((__int64)&v53);
  if ( v6 == (unsigned int)sub_A71AE0(&v53) - 86 <= 0xA )
    goto LABEL_4;
  sub_A759D0((__int64)v54, &v53, 0);
  v45 = (__m128i *)sub_2241130(v54, 0, 0, "Attribute '", 11);
  v56 = &v58;
  if ( (__m128i *)v45->m128i_i64[0] == &v45[1] )
  {
    v58 = _mm_loadu_si128(v45 + 1);
  }
  else
  {
    v56 = (__m128i *)v45->m128i_i64[0];
    v58.m128i_i64[0] = v45[1].m128i_i64[0];
  }
  v46 = v45->m128i_i64[1];
  v45[1].m128i_i8[0] = 0;
  v57 = v46;
  v45->m128i_i64[0] = (__int64)v45[1].m128i_i64;
  v45->m128i_i64[1] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v57) <= 0x18 )
    sub_4262D8((__int64)"basic_string::append");
  v47 = (__m128i *)sub_2241490(&v56, "' should have an Argument", 25, v46);
  v59[0] = &v60;
  if ( (__m128i *)v47->m128i_i64[0] == &v47[1] )
  {
    v60 = _mm_loadu_si128(v47 + 1);
  }
  else
  {
    v59[0] = v47->m128i_i64[0];
    v60.m128i_i64[0] = v47[1].m128i_i64[0];
  }
  v48 = v47->m128i_i64[1];
  v47[1].m128i_i8[0] = 0;
  v59[1] = v48;
  v47->m128i_i64[0] = (__int64)v47[1].m128i_i64;
  v47->m128i_i64[1] = 0;
  v64 = 260;
  v61[0] = v59;
  sub_BE7760(a1, (__int64)v61, &v51);
  if ( (__m128i *)v59[0] != &v60 )
    j_j___libc_free_0(v59[0], v60.m128i_i64[0] + 1);
  if ( v56 != &v58 )
    j_j___libc_free_0(v56, v58.m128i_i64[0] + 1);
  if ( (__int64 *)v54[0] != &v55 )
    j_j___libc_free_0(v54[0], v55 + 1);
}
