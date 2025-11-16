// Function: sub_6B7D60
// Address: 0x6b7d60
//
__int64 __fastcall sub_6B7D60(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  char v6; // dl
  __int64 v7; // rdx
  __int64 v8; // rax
  char v9; // r12
  char v10; // r15
  char v11; // r15
  __int64 v12; // rdi
  __int64 v13; // r15
  char v14; // dl
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rax
  char v18; // dl
  __int64 v19; // rbx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r9
  int v24; // ecx
  __int64 v25; // r12
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // r13
  __int64 v29; // rax
  _BYTE *v30; // rax
  __int64 v31; // rax
  char v32; // cl
  __int64 v33; // rdx
  __int64 v34; // rcx
  char v35; // [rsp+Fh] [rbp-3D1h]
  int v36; // [rsp+10h] [rbp-3D0h]
  unsigned int v37; // [rsp+14h] [rbp-3CCh]
  _BYTE *v38; // [rsp+18h] [rbp-3C8h]
  __int64 v39; // [rsp+20h] [rbp-3C0h]
  _QWORD *v40; // [rsp+28h] [rbp-3B8h]
  unsigned int v41; // [rsp+30h] [rbp-3B0h] BYREF
  int v42; // [rsp+34h] [rbp-3ACh] BYREF
  __int64 v43; // [rsp+38h] [rbp-3A8h] BYREF
  __int64 v44; // [rsp+40h] [rbp-3A0h] BYREF
  __int64 v45; // [rsp+48h] [rbp-398h] BYREF
  _BYTE v46[160]; // [rsp+50h] [rbp-390h] BYREF
  __m128i v47; // [rsp+F0h] [rbp-2F0h] BYREF
  char v48; // [rsp+100h] [rbp-2E0h]
  __int64 v49; // [rsp+180h] [rbp-260h]
  __m128i v50[25]; // [rsp+250h] [rbp-190h] BYREF

  v4 = a1;
  v36 = a2;
  if ( a1 )
  {
    v39 = unk_4D03C40;
    v5 = *(_QWORD *)a1;
    if ( *(_BYTE *)(*(_QWORD *)a1 + 24LL) == 1 )
    {
      do
      {
        while ( 1 )
        {
          v6 = *(_BYTE *)(v5 + 56);
          if ( v6 != 91 )
            break;
          v5 = *(_QWORD *)(*(_QWORD *)(v5 + 72) + 16LL);
          if ( *(_BYTE *)(v5 + 24) != 1 )
            goto LABEL_8;
        }
      }
      while ( v6 == 25 && *(_BYTE *)(v5 + 24) == 1 );
    }
LABEL_8:
    unk_4D03C40 = v5;
    v38 = 0;
  }
  else
  {
    sub_7B8B50(0, a2, a3, a4);
    a1 = 27;
    sub_7BE280(27, 125, 0, 0);
    v39 = 0;
    v32 = *(_BYTE *)(qword_4F061C8 + 52LL);
    a2 = qword_4F061C8 + 52LL;
    *(_BYTE *)(qword_4F061C8 + 52LL) = 0;
    v38 = (_BYTE *)a2;
    v35 = v32;
  }
  v37 = sub_687860(a1, a2);
  sub_68B050(v37, (__int64)&v41, &v44);
  sub_6E1DD0(&v43);
  sub_6E2140(5, v46, 0, qword_4F06BC0 != 0, v4);
  sub_6E2170(v43);
  v7 = qword_4D03C50;
  v8 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  *(_WORD *)(qword_4D03C50 + 17LL) |= 0x2020u;
  v9 = *(_BYTE *)(v7 + 18) >> 7;
  v10 = *(_BYTE *)(v8 + 13);
  *(_BYTE *)(v8 + 13) = v10 | 1;
  v11 = v10 & 1;
  if ( (*(_BYTE *)(v8 + 12) & 0x10) != 0 )
    *(_BYTE *)(v7 + 18) |= 0x80u;
  if ( v4 )
  {
    sub_6F8800(*(_QWORD *)v4, v4, &v47);
    v40 = 0;
  }
  else
  {
    sub_7296C0(v50);
    v40 = (_QWORD *)sub_869D30();
    sub_729730(v50[0].m128i_u32[0]);
    ++*(_BYTE *)(qword_4F061C8 + 36LL);
    v29 = qword_4D03C50;
    ++*(_QWORD *)(qword_4D03C50 + 40LL);
    *(_BYTE *)(v29 + 20) |= 8u;
    sub_69ED20((__int64)&v47, 0, 0, 0);
  }
  v12 = (__int64)&v47;
  *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 13) = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 13)
                                                            & 0xFE
                                                            | v11;
  *(_BYTE *)(qword_4D03C50 + 18LL) = *(_BYTE *)(qword_4D03C50 + 18LL) & 0x7F | (v9 << 7);
  sub_6F6C80(&v47);
  v13 = v47.m128i_i64[0];
  v14 = *(_BYTE *)(v47.m128i_i64[0] + 140);
  if ( v14 == 12 )
  {
    v15 = v47.m128i_i64[0];
    do
    {
      v15 = *(_QWORD *)(v15 + 160);
      v14 = *(_BYTE *)(v15 + 140);
    }
    while ( v14 == 12 );
  }
  if ( !v14 )
  {
    v16 = (__int64)qword_4F04C68;
    *(_BYTE *)(qword_4D03C50 + 18LL) &= ~0x20u;
    v17 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    v18 = *(_BYTE *)(v17 + 12);
    if ( (v18 & 0x10) != 0 )
      *(_BYTE *)(v17 + 12) = v18 | 0x40;
    goto LABEL_19;
  }
  if ( !v4 || (*(_DWORD *)(v4 + 40) & 0x86140) != 0 )
  {
    v13 = sub_7259C0(12);
    v21 = sub_6968F0((__int64)&v47, &v42);
    v24 = *(_DWORD *)(v13 + 184);
    *(_QWORD *)(v13 + 160) = v21;
    v25 = v21;
    *(_DWORD *)(v13 + 184) = v24 & 0xFFFDFF00 | (v42 << 17) & 0x20000 | 1;
    if ( (dword_4F04C44 != -1
       || (v26 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v26 + 6) & 6) != 0)
       || *(_BYTE *)(v26 + 4) == 12)
      && ((unsigned int)sub_696840((__int64)&v47) || *(_QWORD *)&dword_4D03B80 == v25) )
    {
      *(_BYTE *)(v13 + 186) |= 8u;
      sub_6F40C0(&v47);
    }
    else if ( dword_4D0425C )
    {
      if ( v48 == 1 )
      {
        v30 = (_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
        if ( v30[4] == 1 && (dword_4F04C44 != -1 || (v30[6] & 6) != 0) && (v30[12] & 0x10) == 0 )
        {
          v31 = sub_724DC0(&v47, &dword_4F04C44, v22, dword_4D0425C, v50, v23);
          v45 = v31;
          if ( *(_BYTE *)(v49 + 24) == 1 && *(_BYTE *)(v49 + 56) == 105 && (unsigned int)sub_719770(v49, v31, 0, 0) )
          {
            sub_68F8E0(v50, &v47);
            sub_6E6A50(v45, &v47);
            sub_6E4BC0(&v47, v50);
          }
          sub_724E30(&v45);
        }
      }
    }
    sub_7296C0(v50);
    v16 = 0;
    v27 = sub_6F6F40(&v47, 0);
    v12 = v50[0].m128i_u32[0];
    v28 = v27;
    sub_729730(v50[0].m128i_u32[0]);
    if ( (*(_BYTE *)(v28 - 8) & 1) != 0 )
    {
      *(_QWORD *)(*(_QWORD *)(v13 + 168) + 24LL) = v28;
    }
    else
    {
      v16 = 6;
      v12 = v28;
      sub_72D910(v28, 6, v13);
      *(_QWORD *)(v13 + 48) = *(_QWORD *)(qword_4F04C68[0] + 776LL * (int)v37 + 216);
    }
LABEL_19:
    if ( !v40 )
      goto LABEL_24;
    goto LABEL_20;
  }
  v16 = (__int64)v50;
  v12 = (__int64)&v47;
  v13 = sub_6968F0((__int64)&v47, v50);
  sub_6E4710(&v47);
  if ( !v40 )
    goto LABEL_25;
LABEL_20:
  if ( *v40 )
  {
    if ( !dword_4F04C3C )
      sub_8699D0(v13, 6, v40);
    v16 = 6;
    v12 = v13;
    sub_869D70(v13, 6);
  }
  else
  {
    v12 = (__int64)v40;
    v16 = (unsigned int)dword_4F04C64;
    sub_869FD0(v40, (unsigned int)dword_4F04C64);
  }
LABEL_24:
  if ( v4 )
  {
LABEL_25:
    unk_4D03C40 = v39;
    goto LABEL_26;
  }
  v16 = 18;
  v12 = 28;
  --*(_BYTE *)(qword_4F061C8 + 36LL);
  --*(_QWORD *)(qword_4D03C50 + 40LL);
  if ( (unsigned int)sub_7BE5B0(28, 18, 0, 0) && !v36 )
    sub_7B8B50(28, 18, v33, v34);
  if ( v38 )
    *v38 = v35;
LABEL_26:
  sub_6E2B30(v12, v16);
  sub_6E1DF0(v43);
  v19 = v44;
  sub_729730(v41);
  qword_4F06BC0 = v19;
  return v13;
}
