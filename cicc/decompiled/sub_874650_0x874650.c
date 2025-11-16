// Function: sub_874650
// Address: 0x874650
//
__int64 __fastcall sub_874650(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int *v3; // rsi
  unsigned int *v4; // rbx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rax
  __int64 v18; // rdx
  int v19; // ecx
  __int64 v20; // rdx
  __int64 v21; // rax
  _QWORD *v22; // r8
  __int64 v23; // r9
  int v24; // r13d
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rdi
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 *v32; // r9
  __int64 result; // rax
  int v34; // r15d
  __int64 v35; // rax
  __int64 *v36; // rcx
  __int64 v37; // r14
  __int64 v38; // rdx
  char v39; // al
  __int64 v40; // rcx
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 *v44; // r9
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 *v52; // r9
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 *v57; // r9
  __int64 v58; // rdi
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 *v62; // r9
  int v63; // eax
  __int64 v64; // rax
  __int64 v65; // rsi
  __int64 v66; // rcx
  int v67; // eax
  __int64 v68; // rax
  __int64 v69; // rax
  int v70; // [rsp+4h] [rbp-38Ch]
  __int64 v71; // [rsp+8h] [rbp-388h]
  __int64 *v72; // [rsp+8h] [rbp-388h]
  __int64 *v73; // [rsp+8h] [rbp-388h]
  __int64 v74; // [rsp+8h] [rbp-388h]
  __int64 v75; // [rsp+8h] [rbp-388h]
  __int64 v76; // [rsp+8h] [rbp-388h]
  int v77; // [rsp+10h] [rbp-380h]
  __int64 v78; // [rsp+10h] [rbp-380h]
  unsigned int v79; // [rsp+10h] [rbp-380h]
  int v80; // [rsp+18h] [rbp-378h]
  char v81; // [rsp+1Fh] [rbp-371h]
  int v82; // [rsp+24h] [rbp-36Ch] BYREF
  __int64 v83; // [rsp+28h] [rbp-368h] BYREF
  __m128i v84; // [rsp+30h] [rbp-360h] BYREF
  _BYTE v85[160]; // [rsp+40h] [rbp-350h] BYREF
  __int64 v86[20]; // [rsp+E0h] [rbp-2B0h] BYREF
  _QWORD v87[66]; // [rsp+180h] [rbp-210h] BYREF

  v82 = 0;
  v81 = 0;
  v83 = *(_QWORD *)&dword_4F063F8;
  v80 = dword_4F5FD80 | qword_4F5FD78;
  if ( dword_4F077C4 != 2 && unk_4F07778 > 199900 )
  {
    sub_86EE70(a1, a2, a3);
    v81 = 1;
  }
  v3 = *(unsigned int **)(qword_4D03B98 + 176LL * unk_4D03B90 + 160);
  if ( !v3 )
    v3 = &dword_4F063F8;
  v4 = (unsigned int *)sub_86E480(0xDu, v3);
  if ( !dword_4F04C3C )
    sub_8699D0((__int64)v4, 21, 0);
  sub_854980(0, (__int64)v4);
  sub_86D170(6, (__int64)v4, 0, 0, v5, v6);
  sub_7B8B50(6u, v4, v7, v8, v9, v10);
  v77 = 0;
  if ( word_4F06418[0] == 269 )
  {
    v84.m128i_i64[1] = *(_QWORD *)&dword_4F063F8;
    sub_7B8B50(6u, v4, v11, v12, v13, v14);
    v77 = 1;
  }
  sub_7BE280(0x1Bu, 125, 0, 0, v13, v14);
  v17 = qword_4F061C8;
  ++*(_BYTE *)(qword_4F061C8 + 36LL);
  if ( dword_4F077C4 != 2 )
  {
    ++*(_BYTE *)(v17 + 83);
    v18 = qword_4D03B98 + 176LL * unk_4D03B90;
    *(_BYTE *)(v18 + 4) |= 0x10u;
    goto LABEL_12;
  }
  if ( (unsigned __int16)sub_67B420() == 75 )
  {
    ++*(_BYTE *)(qword_4F061C8 + 83LL);
    v64 = 176LL * unk_4D03B90;
    v18 = v64 + qword_4D03B98;
    *(_BYTE *)(v64 + qword_4D03B98 + 4) |= 0x10u;
    if ( dword_4F077C4 == 2 )
    {
      v75 = v18;
      v65 = v64 + qword_4D03B98;
      v66 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 336);
      *(_BYTE *)(v65 + 5) |= 0x80u;
      *(_QWORD *)(v65 + 144) = v66;
      *(_QWORD *)(qword_4D03B98 + v64 + 136) = v87;
      v87[0] = 0;
      v67 = sub_679C10(2u);
      v18 = v75;
      if ( v67 )
      {
LABEL_62:
        if ( dword_4F077C4 == 2 && !unk_4D04350 )
        {
          v76 = *(_QWORD *)(*(_QWORD *)(v18 + 8) + 80LL);
          *(_QWORD *)(v76 + 16) = sub_8602B0((__int64)v85);
          v68 = sub_86B2C0(0);
          *(_QWORD *)(v68 + 24) = *(_QWORD *)&dword_4F063F8;
          *(_QWORD *)(v68 + 56) = qword_4F06BC0;
          sub_86CBE0(v68);
        }
        sub_86E660(0, 0);
LABEL_15:
        v19 = dword_4F077C4;
        v20 = 176LL * unk_4D03B90;
        v21 = v20 + qword_4D03B98;
        if ( dword_4F077C4 == 2 )
        {
          *(_BYTE *)(v21 + 5) &= ~0x80u;
          *(_QWORD *)(v21 + 144) = 0;
          v21 = v20 + qword_4D03B98;
          *(_QWORD *)(v20 + qword_4D03B98 + 136) = 0;
        }
        *(_BYTE *)(v21 + 4) &= ~0x10u;
        *(_QWORD *)(v21 + 48) = 0;
        *(_QWORD *)(v21 + 56) = 0;
        --*(_BYTE *)(qword_4F061C8 + 83LL);
        if ( v19 != 2 )
          goto LABEL_18;
        goto LABEL_37;
      }
      if ( dword_4F077C4 == 2 )
      {
        if ( word_4F06418[0] == 75 )
          goto LABEL_14;
        goto LABEL_34;
      }
    }
LABEL_12:
    if ( unk_4F07778 > 199900 )
    {
      v74 = v18;
      v63 = sub_651B00(3u);
      v18 = v74;
      if ( v63 )
        goto LABEL_62;
    }
    if ( word_4F06418[0] == 75 )
    {
LABEL_14:
      sub_7BE280(0x4Bu, 65, 0, 0, v15, v16);
      goto LABEL_15;
    }
LABEL_34:
    sub_86E9F0(0);
    goto LABEL_14;
  }
  if ( dword_4F077C4 != 2 )
  {
LABEL_18:
    if ( v77 )
      sub_6851C0(0xBD7u, &v84.m128i_i32[2]);
    ++*(_BYTE *)(qword_4F061C8 + 83LL);
    if ( word_4F06418[0] != 75 )
      sub_86E100((__int64)v4, &v82);
    sub_7BE280(0x4Bu, 65, 0, 0, v15, v16);
    v24 = 0;
    --*(_BYTE *)(qword_4F061C8 + 83LL);
    if ( word_4F06418[0] != 28 )
    {
      v70 = dword_4F5FD80;
      v71 = qword_4F5FD78;
      v34 = unk_4D048C4;
      v78 = *((_QWORD *)v4 + 10);
      qword_4F5FD78 = 0x100000001LL;
      dword_4F5FD80 = 0;
      unk_4D048C4 = 1;
      v35 = sub_6B9820(1u, 0, 0, 0, 0);
      v23 = v78;
      unk_4D048C4 = v34;
      *(_QWORD *)(v78 + 8) = v35;
      qword_4F5FD78 = v71;
      dword_4F5FD80 = v70;
    }
    goto LABEL_24;
  }
LABEL_37:
  if ( (unsigned __int16)sub_67B420() != 55 )
    goto LABEL_18;
  v36 = (__int64 *)*((_QWORD *)v4 + 10);
  if ( !unk_4F07728 )
  {
    v72 = (__int64 *)*((_QWORD *)v4 + 10);
    sub_684AA0((_DWORD)qword_4F077B4 == 0 ? 8 : 5, 0xD5Du, &dword_4F063F8);
    v36 = v72;
  }
  v73 = v36;
  *(_DWORD *)(qword_4D03B98 + 176LL * unk_4D03B90) = 7;
  *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 8) &= ~8u;
  sub_7268E0((__int64)v4, 14);
  v37 = *((_QWORD *)v4 + 10);
  v38 = *v73;
  v39 = *(_BYTE *)(v37 + 72);
  *(_QWORD *)v37 = *v73;
  v40 = v73[2];
  *(_QWORD *)(v37 + 24) = v40;
  *(_BYTE *)(v37 + 72) = v77 | v39 & 0xFE;
  if ( v38 )
  {
    if ( *(_BYTE *)(v38 + 40) != 11 || *(_DWORD *)v38 )
      v87[0] = *(_QWORD *)v38;
    else
      v87[0] = **(_QWORD **)(v38 + 72);
    if ( unk_4D041B0 )
    {
      if ( dword_4F077BC && (dword_4F077C4 != 2 || unk_4F07778 <= 202001) && !dword_4F5FD58 )
      {
        if ( sub_729F80(dword_4F063F8) )
        {
          v40 = *(_QWORD *)(v37 + 24);
        }
        else
        {
          sub_684B30(0xBD6u, v87);
          v40 = *(_QWORD *)(v37 + 24);
          dword_4F5FD58 = 1;
        }
      }
    }
    else
    {
      sub_6851C0(0xBD6u, v87);
      v40 = *(_QWORD *)(v37 + 24);
    }
  }
  if ( !v40 )
    *(_QWORD *)(v37 + 24) = sub_86D550((__int64)v85);
  *(_QWORD *)(v37 + 32) = sub_86D550((__int64)v86);
  v41 = sub_86B2C0(0);
  *(_QWORD *)(v41 + 24) = *(_QWORD *)&dword_4F063F8;
  *(_QWORD *)(v41 + 56) = qword_4F06BC0;
  sub_86CBE0(v41);
  ++*(_BYTE *)(qword_4F061C8 + 63LL);
  memset(v87, 0, 0x1D8u);
  v87[19] = v87;
  v87[3] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v87[22]) |= 1u;
  BYTE2(v87[16]) |= 0x20u;
  sub_662DE0((unsigned int *)v87, 0);
  sub_64FF30((__int64)v87);
  if ( v87[0] )
  {
    if ( *(_BYTE *)(v87[0] + 80LL) == 7 )
    {
      v69 = *(_QWORD *)(v87[0] + 88LL);
      *(_QWORD *)(v37 + 8) = v69;
      if ( v69 )
        *(_BYTE *)(v69 + 175) |= 0x80u;
    }
  }
  sub_863FA0(0, 0, v42, v43, (__int64)v87, v44);
  sub_7BE280(0x37u, 53, 0, 0, v45, v46);
  --*(_BYTE *)(qword_4F061C8 + 63LL);
  v79 = dword_4F06650[0];
  sub_6D08F0((__int64)v4, &v84);
  sub_699930((__int64)v4, &v84, v79, (__int64)v86);
  sub_8601E0(*(_QWORD *)(v37 + 32), v86);
  v22 = v87;
  if ( (v87[16] & 0x10000000LL) != 0 )
    sub_6570B0((__int64)v87);
  v24 = 1;
LABEL_24:
  v25 = 18;
  sub_7BE280(0x1Cu, 18, 0, 0, (__int64)v22, v23);
  --*(_BYTE *)(qword_4F061C8 + 36LL);
  sub_8745D0();
  v26 = qword_4D03B98 + 176LL * unk_4D03B90;
  if ( !v80 && (*(_BYTE *)(v26 + 5) & 0x60) == 0 )
  {
    v25 = (__int64)&v83;
    sub_684B30(0x80u, &v83);
    dword_4F5FD80 = 1;
    v26 = qword_4D03B98 + 176LL * unk_4D03B90;
  }
  v27 = *(_QWORD *)(v26 + 80);
  if ( v27 )
  {
    v25 = *(_QWORD *)(v26 + 88);
    sub_86EEF0(v27, v25);
  }
  if ( v82 )
  {
    v58 = sub_86B2C0(5);
    sub_86CBE0(v58);
    sub_863FC0(v58, v25, v59, v60, v61, v62);
  }
  if ( v24 )
  {
    v47 = sub_86B2C0(5);
    sub_86CBE0(v47);
    v48 = sub_86B2C0(5);
    sub_86CBE0(v48);
    sub_863FA0(1, v25, v49, v50, v51, v52);
    v53 = sub_86B2C0(5);
    sub_86CBE0(v53);
    sub_863FA0(1, v25, v54, v55, v56, v57);
  }
  else if ( *(_QWORD *)(*((_QWORD *)v4 + 10) + 16LL) )
  {
    v28 = sub_86B2C0(5);
    sub_86CBE0(v28);
    sub_863FC0(v28, v25, v29, v30, v31, v32);
  }
  sub_86F030();
  sub_86C020((__int64)v4);
  result = *(_QWORD *)&dword_4F061D8;
  *((_QWORD *)v4 + 1) = *(_QWORD *)&dword_4F061D8;
  if ( v81 )
    return sub_86F430(*(_BYTE **)(qword_4D03B98 + 176LL * unk_4D03B90 + 8));
  return result;
}
