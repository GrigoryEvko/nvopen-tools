// Function: sub_1BF2D30
// Address: 0x1bf2d30
//
__int64 __fastcall sub_1BF2D30(__int64 *a1)
{
  __int64 v2; // r12
  __int64 *v3; // rbx
  __int64 *v4; // rax
  __int64 v5; // r12
  __int64 v6; // r14
  __int64 v7; // r12
  unsigned __int8 v8; // al
  __int64 v9; // rsi
  __int64 *v10; // rax
  __int64 *v11; // rdi
  __int64 *v12; // rcx
  __int64 v13; // rax
  int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r14
  _QWORD *v18; // r13
  char *v19; // rax
  _QWORD *v20; // rbx
  _QWORD *v21; // r12
  _QWORD *v22; // rdi
  unsigned __int64 v23; // rdi
  __int64 *v24; // rax
  __int64 *v26; // r14
  __int64 v27; // r13
  unsigned __int64 v28; // rbx
  __int64 v29; // rax
  _QWORD *v30; // rcx
  __int64 v31; // rdx
  __int64 v32; // r12
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 *v35; // rbx
  __int64 *v36; // r14
  __int64 v37; // rax
  _QWORD *v38; // r14
  unsigned __int64 v39; // rax
  __int64 v40; // r13
  __int64 v41; // rbx
  char *v42; // rax
  _QWORD *v43; // rbx
  _QWORD *v44; // rdi
  _QWORD *v45; // r14
  unsigned __int64 v46; // rax
  __int64 v47; // r13
  __int64 v48; // rbx
  char *v49; // rax
  _QWORD *v50; // rbx
  _QWORD *v51; // rdi
  _QWORD *v52; // r14
  char *v53; // rax
  _QWORD *v54; // rbx
  _QWORD *v55; // r12
  _QWORD *v56; // rdi
  unsigned __int8 v57; // [rsp+Fh] [rbp-2A1h]
  __int64 v58; // [rsp+10h] [rbp-2A0h]
  __int64 v59; // [rsp+18h] [rbp-298h]
  __int64 *v60; // [rsp+20h] [rbp-290h]
  __int64 *v61; // [rsp+28h] [rbp-288h]
  __int64 *i; // [rsp+28h] [rbp-288h]
  __int64 v63; // [rsp+30h] [rbp-280h] BYREF
  __int64 *v64; // [rsp+38h] [rbp-278h]
  __int64 *v65; // [rsp+40h] [rbp-270h]
  __int64 v66; // [rsp+48h] [rbp-268h]
  int v67; // [rsp+50h] [rbp-260h]
  _BYTE v68[72]; // [rsp+58h] [rbp-258h] BYREF
  _QWORD v69[11]; // [rsp+A0h] [rbp-210h] BYREF
  _QWORD *v70; // [rsp+F8h] [rbp-1B8h]
  unsigned int v71; // [rsp+100h] [rbp-1B0h]
  _BYTE v72[424]; // [rsp+108h] [rbp-1A8h] BYREF

  v2 = *a1;
  v57 = byte_4FB9A80;
  if ( !byte_4FB9A80 )
  {
    v52 = (_QWORD *)a1[7];
    v53 = sub_1BF18B0(a1[58]);
    sub_1BF1750((__int64)v69, (__int64)v53, (__int64)"IfConversionDisabled", 20, v2, 0);
    sub_15CAB20((__int64)v69, "if-conversion is disabled", 0x19u);
    sub_143AA50(v52, (__int64)v69);
    v54 = v70;
    v69[0] = &unk_49ECF68;
    v55 = &v70[11 * v71];
    if ( v70 != v55 )
    {
      do
      {
        v55 -= 11;
        v56 = (_QWORD *)v55[4];
        if ( v56 != v55 + 6 )
          j_j___libc_free_0(v56, v55[6] + 1LL);
        if ( (_QWORD *)*v55 != v55 + 2 )
          j_j___libc_free_0(*v55, v55[2] + 1LL);
      }
      while ( v54 != v55 );
      v55 = v70;
    }
    if ( v55 != (_QWORD *)v72 )
      _libc_free((unsigned __int64)v55);
    return v57;
  }
  v67 = 0;
  v3 = *(__int64 **)(v2 + 32);
  v64 = (__int64 *)v68;
  v65 = (__int64 *)v68;
  v4 = *(__int64 **)(v2 + 40);
  v63 = 0;
  v66 = 8;
  v61 = v4;
  if ( v3 == v4 )
    return v57;
  do
  {
    v5 = *v3;
    if ( !(unsigned __int8)sub_1BF29F0(a1, *v3) )
    {
      v6 = *(_QWORD *)(v5 + 48);
      v7 = v5 + 40;
      if ( v7 != v6 )
      {
        while ( 1 )
        {
          if ( !v6 )
            BUG();
          v8 = *(_BYTE *)(v6 - 8);
          if ( v8 <= 0x17u )
            goto LABEL_10;
          if ( v8 == 54 || v8 == 55 )
            break;
          if ( v8 != 78 )
            goto LABEL_10;
          v13 = *(_QWORD *)(v6 - 48);
          if ( *(_BYTE *)(v13 + 16) )
            goto LABEL_10;
          v14 = *(_DWORD *)(v13 + 36);
          if ( v14 == 4085 || v14 == 4057 )
          {
            v15 = 1;
            v16 = *(_DWORD *)(v6 - 4) & 0xFFFFFFF;
          }
          else
          {
            if ( v14 != 4503 && v14 != 4492 )
              goto LABEL_10;
            v15 = 2;
            v16 = *(_DWORD *)(v6 - 4) & 0xFFFFFFF;
          }
          v9 = *(_QWORD *)(v6 + 24 * (v15 - v16) - 24);
          if ( v9 )
            goto LABEL_15;
LABEL_10:
          v6 = *(_QWORD *)(v6 + 8);
          if ( v7 == v6 )
            goto LABEL_4;
        }
        v9 = *(_QWORD *)(v6 - 48);
        if ( !v9 )
          goto LABEL_10;
LABEL_15:
        v10 = v64;
        if ( v65 != v64 )
          goto LABEL_16;
        v11 = &v64[HIDWORD(v66)];
        if ( v64 != v11 )
        {
          v12 = 0;
          while ( *v10 != v9 )
          {
            if ( *v10 == -2 )
              v12 = v10;
            if ( v11 == ++v10 )
            {
              if ( !v12 )
                goto LABEL_47;
              *v12 = v9;
              --v67;
              ++v63;
              goto LABEL_10;
            }
          }
          goto LABEL_10;
        }
LABEL_47:
        if ( HIDWORD(v66) < (unsigned int)v66 )
        {
          ++HIDWORD(v66);
          *v11 = v9;
          ++v63;
        }
        else
        {
LABEL_16:
          sub_16CCBA0((__int64)&v63, v9);
        }
        goto LABEL_10;
      }
    }
LABEL_4:
    ++v3;
  }
  while ( v61 != v3 );
  v26 = *(__int64 **)(*a1 + 32);
  v27 = *v26;
  v60 = *(__int64 **)(*a1 + 40);
  v58 = *v26;
  if ( v60 == v26 )
  {
LABEL_83:
    v23 = (unsigned __int64)v65;
    v24 = v64;
    goto LABEL_43;
  }
  for ( i = v26 + 1; ; ++i )
  {
    v28 = sub_157EBA0(v27);
    if ( *(_BYTE *)(v28 + 16) != 26 )
    {
      v17 = *a1;
      v18 = (_QWORD *)a1[7];
      v19 = sub_1BF18B0(a1[58]);
      sub_1BF1750((__int64)v69, (__int64)v19, (__int64)"LoopContainsSwitch", 18, v17, v28);
      sub_15CAB20((__int64)v69, "loop contains a switch statement", 0x20u);
      sub_143AA50(v18, (__int64)v69);
      v20 = v70;
      v69[0] = &unk_49ECF68;
      v21 = &v70[11 * v71];
      if ( v70 == v21 )
        goto LABEL_40;
      do
      {
        v21 -= 11;
        v22 = (_QWORD *)v21[4];
        if ( v22 != v21 + 6 )
          j_j___libc_free_0(v22, v21[6] + 1LL);
        if ( (_QWORD *)*v21 != v21 + 2 )
          j_j___libc_free_0(*v21, v21[2] + 1LL);
      }
      while ( v20 != v21 );
      goto LABEL_39;
    }
    if ( (unsigned __int8)sub_1BF29F0(a1, v27) )
      break;
    if ( v58 != v27 )
    {
      v29 = sub_157F280(v27);
      v59 = v31;
      v32 = v29;
      if ( v29 != v31 )
      {
        while ( 1 )
        {
          v33 = 24LL * (*(_DWORD *)(v32 + 20) & 0xFFFFFFF);
          if ( (*(_BYTE *)(v32 + 23) & 0x40) != 0 )
          {
            v34 = *(_QWORD *)(v32 - 8);
            v35 = (__int64 *)(v34 + v33);
          }
          else
          {
            v35 = (__int64 *)v32;
            v34 = v32 - v33;
          }
          v36 = (__int64 *)v34;
          if ( v35 != (__int64 *)v34 )
            break;
LABEL_61:
          v37 = *(_QWORD *)(v32 + 32);
          if ( !v37 )
            BUG();
          v32 = 0;
          if ( *(_BYTE *)(v37 - 8) == 77 )
            v32 = v37 - 24;
          if ( v59 == v32 )
            goto LABEL_65;
        }
        while ( *(_BYTE *)(*v36 + 16) > 0x10u || !(unsigned __int8)sub_1593DF0(*v36, v27, v34, v30) )
        {
          v36 += 3;
          if ( v35 == v36 )
            goto LABEL_61;
        }
        v45 = (_QWORD *)a1[7];
        v46 = sub_157EBA0(v27);
        v47 = *a1;
        v48 = v46;
        v49 = sub_1BF18B0(a1[58]);
        sub_1BF1750((__int64)v69, (__int64)v49, (__int64)"NoCFGForSelect", 14, v47, v48);
        sub_15CAB20((__int64)v69, "control flow cannot be substituted for a select", 0x2Fu);
        sub_143AA50(v45, (__int64)v69);
        v50 = v70;
        v69[0] = &unk_49ECF68;
        v21 = &v70[11 * v71];
        if ( v70 != v21 )
        {
          do
          {
            v21 -= 11;
            v51 = (_QWORD *)v21[4];
            if ( v51 != v21 + 6 )
              j_j___libc_free_0(v51, v21[6] + 1LL);
            if ( (_QWORD *)*v21 != v21 + 2 )
              j_j___libc_free_0(*v21, v21[2] + 1LL);
          }
          while ( v50 != v21 );
LABEL_39:
          v21 = v70;
          goto LABEL_40;
        }
        goto LABEL_40;
      }
    }
LABEL_65:
    if ( v60 == i )
      goto LABEL_83;
    v27 = *i;
  }
  if ( (unsigned __int8)sub_1BF2A10((__int64)a1, v27, (__int64)&v63) )
    goto LABEL_65;
  v38 = (_QWORD *)a1[7];
  v39 = sub_157EBA0(v27);
  v40 = *a1;
  v41 = v39;
  v42 = sub_1BF18B0(a1[58]);
  sub_1BF1750((__int64)v69, (__int64)v42, (__int64)"NoCFGForSelect", 14, v40, v41);
  sub_15CAB20((__int64)v69, "control flow cannot be substituted for a select", 0x2Fu);
  sub_143AA50(v38, (__int64)v69);
  v43 = v70;
  v69[0] = &unk_49ECF68;
  v21 = &v70[11 * v71];
  if ( v70 != v21 )
  {
    do
    {
      v21 -= 11;
      v44 = (_QWORD *)v21[4];
      if ( v44 != v21 + 6 )
        j_j___libc_free_0(v44, v21[6] + 1LL);
      if ( (_QWORD *)*v21 != v21 + 2 )
        j_j___libc_free_0(*v21, v21[2] + 1LL);
    }
    while ( v43 != v21 );
    goto LABEL_39;
  }
LABEL_40:
  if ( v21 != (_QWORD *)v72 )
    _libc_free((unsigned __int64)v21);
  v57 = 0;
  v23 = (unsigned __int64)v65;
  v24 = v64;
LABEL_43:
  if ( (__int64 *)v23 != v24 )
    _libc_free(v23);
  return v57;
}
