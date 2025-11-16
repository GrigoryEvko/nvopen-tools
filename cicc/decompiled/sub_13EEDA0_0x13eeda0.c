// Function: sub_13EEDA0
// Address: 0x13eeda0
//
__int64 __fastcall sub_13EEDA0(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4, int *a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rcx
  unsigned int v9; // r8d
  __int64 v10; // r15
  int v11; // eax
  unsigned int v12; // r13d
  __int64 v13; // rax
  __int64 v15; // rax
  unsigned __int8 *v16; // r13
  int v17; // eax
  __int64 v18; // r11
  __int64 v19; // r10
  char v20; // al
  int v21; // eax
  __int64 v22; // rdx
  unsigned __int32 v23; // r15d
  __int64 v24; // r13
  char v25; // bl
  unsigned int v26; // r13d
  __int64 v27; // rbx
  int v28; // eax
  __int64 v29; // rdx
  __int64 v30; // rsi
  unsigned __int32 v31; // eax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int32 v34; // eax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  unsigned int v38; // eax
  unsigned int v39; // eax
  char v40; // bl
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rcx
  unsigned int v44; // r8d
  __int64 v45; // r10
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // r15
  unsigned int v50; // eax
  unsigned __int8 *v51; // rdx
  unsigned __int8 *v52; // rbx
  unsigned __int8 *v53; // r9
  __int64 *v54; // rax
  unsigned __int8 *v56; // [rsp+20h] [rbp-140h]
  __int64 v59; // [rsp+40h] [rbp-120h]
  __int64 v60; // [rsp+40h] [rbp-120h]
  __int64 v61; // [rsp+48h] [rbp-118h]
  char v63; // [rsp+60h] [rbp-100h]
  unsigned __int8 v64; // [rsp+68h] [rbp-F8h]
  __int64 v65; // [rsp+70h] [rbp-F0h] BYREF
  unsigned __int32 v66; // [rsp+78h] [rbp-E8h]
  unsigned __int64 v67; // [rsp+80h] [rbp-E0h] BYREF
  unsigned int v68; // [rsp+88h] [rbp-D8h]
  unsigned __int64 v69; // [rsp+90h] [rbp-D0h]
  unsigned int v70; // [rsp+98h] [rbp-C8h]
  int v71; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v72; // [rsp+A8h] [rbp-B8h] BYREF
  unsigned __int32 v73; // [rsp+B0h] [rbp-B0h]
  __int64 v74; // [rsp+B8h] [rbp-A8h] BYREF
  unsigned int v75; // [rsp+C0h] [rbp-A0h]
  unsigned __int64 v76; // [rsp+D0h] [rbp-90h] BYREF
  __int64 v77; // [rsp+D8h] [rbp-88h] BYREF
  unsigned __int64 v78; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v79; // [rsp+E8h] [rbp-78h]
  unsigned int v80; // [rsp+F0h] [rbp-70h]
  __m128i v81; // [rsp+100h] [rbp-60h] BYREF
  unsigned __int64 v82; // [rsp+110h] [rbp-50h]
  __int64 v83; // [rsp+118h] [rbp-48h] BYREF
  unsigned int v84; // [rsp+120h] [rbp-40h]

  v71 = 0;
  v7 = sub_157EBA0(a3);
  v10 = v7;
  if ( *(_BYTE *)(v7 + 16) != 26 )
    goto LABEL_20;
  if ( (*(_DWORD *)(v7 + 20) & 0xFFFFFFF) != 3 )
    goto LABEL_3;
  v15 = *(_QWORD *)(v7 - 24);
  if ( v15 == *(_QWORD *)(v10 - 48) )
    goto LABEL_3;
  v16 = *(unsigned __int8 **)(v10 - 72);
  v64 = a4 == v15;
  if ( a2 == v16 )
  {
    v46 = sub_16498A0(a2);
    v47 = sub_1643320(v46);
    v48 = sub_159C470(v47, v64, 0);
    v81.m128i_i32[0] = 0;
    if ( *(_BYTE *)(v48 + 16) != 9 )
      sub_13EA740(v81.m128i_i32, v48);
    sub_13E8810(&v71, (unsigned int *)&v81);
    sub_13EA000((__int64)&v81);
    v11 = v71;
    goto LABEL_5;
  }
  sub_13EE900(v81.m128i_i32, (__int64)a2, (__int64)v16, v64);
  sub_13E8810(&v71, (unsigned int *)&v81);
  if ( v81.m128i_i32[0] == 3 )
  {
    if ( v84 > 0x40 && v83 )
      j_j___libc_free_0_0(v83);
    if ( (unsigned int)v82 > 0x40 && v81.m128i_i64[1] )
      j_j___libc_free_0_0(v81.m128i_i64[1]);
  }
  v11 = v71;
  if ( v71 != 4 )
    goto LABEL_5;
  v17 = a2[16];
  if ( (unsigned __int8)v17 <= 0x17u
    || *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 11
    || (unsigned int)(v17 - 60) > 0xC && (unsigned int)(v17 - 35) > 0x11 )
  {
    goto LABEL_19;
  }
  v41 = sub_157EB90(a4);
  v60 = sub_1632FA0(v41);
  if ( (unsigned __int8)sub_13E7650(a2, (__int64)v16, v42, v43, v44) )
  {
    LODWORD(v77) = 1;
    v76 = v64;
    sub_13EA5B0(v81.m128i_i32, (__int64)a2, v16, (__int64)&v76, v45);
    sub_13E8810(&v71, (unsigned int *)&v81);
    sub_13EA000((__int64)&v81);
    sub_135E100((__int64 *)&v76);
    goto LABEL_156;
  }
  v49 = 0;
  v50 = *((_DWORD *)a2 + 5) & 0xFFFFFFF;
  if ( !v50 )
    goto LABEL_156;
  while ( 1 )
  {
    v51 = (a2[23] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)a2 - 1) : &a2[-24 * v50];
    v52 = *(unsigned __int8 **)&v51[24 * v49];
    sub_13EE900((int *)&v76, (__int64)v52, (__int64)v16, v64);
    if ( (_DWORD)v76 != 1 )
      break;
    if ( *(_BYTE *)(v77 + 16) != 13 )
      goto LABEL_192;
    LOBYTE(v69) = 1;
    v68 = *(_DWORD *)(v77 + 32);
    if ( v68 <= 0x40 )
    {
      v53 = v52;
      v67 = *(_QWORD *)(v77 + 24);
LABEL_198:
      sub_13EA5B0(v81.m128i_i32, (__int64)a2, v53, (__int64)&v67, v60);
      goto LABEL_199;
    }
    sub_16A4FD0(&v67, v77 + 24);
LABEL_207:
    if ( (_BYTE)v69 )
    {
      v53 = v52;
      goto LABEL_198;
    }
LABEL_192:
    ++v49;
    sub_13EA000((__int64)&v76);
    v50 = *((_DWORD *)a2 + 5) & 0xFFFFFFF;
    if ( v50 <= (unsigned int)v49 )
      goto LABEL_156;
  }
  if ( (_DWORD)v76 != 3 || !sub_13E9F20(&v77) )
    goto LABEL_192;
  v54 = sub_13E9F20(&v77);
  LOBYTE(v69) = 1;
  v68 = *((_DWORD *)v54 + 2);
  if ( v68 > 0x40 )
  {
    sub_16A4FD0(&v67, v54);
    goto LABEL_207;
  }
  v67 = *v54;
  sub_13EA5B0(v81.m128i_i32, (__int64)a2, v52, (__int64)&v67, v60);
LABEL_199:
  sub_13E8810(&v71, (unsigned int *)&v81);
  sub_13EA000((__int64)&v81);
  if ( (_BYTE)v69 )
    sub_135E100((__int64 *)&v67);
  sub_13EA000((__int64)&v76);
LABEL_156:
  v11 = v71;
  if ( v71 != 4 )
    goto LABEL_5;
LABEL_19:
  v10 = sub_157EBA0(a3);
LABEL_20:
  if ( *(_BYTE *)(v10 + 16) != 27 )
    goto LABEL_3;
  if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
    v18 = *(_QWORD *)(v10 - 8);
  else
    v18 = v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF);
  v19 = *(_QWORD *)a2;
  v56 = *(unsigned __int8 **)v18;
  v20 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
  if ( !*(_QWORD *)v18 )
  {
    if ( v20 != 11 )
      goto LABEL_3;
    goto LABEL_26;
  }
  if ( v20 != 11 )
    goto LABEL_3;
  if ( a2 == *(unsigned __int8 **)v18 )
  {
    v63 = 0;
  }
  else
  {
LABEL_26:
    v21 = a2[16];
    if ( (unsigned __int8)v21 <= 0x17u )
      goto LABEL_3;
    v22 = (unsigned int)(v21 - 60);
    if ( (unsigned int)v22 > 0xC && (unsigned int)(v21 - 35) > 0x11 )
      goto LABEL_3;
    v63 = sub_13E7650(a2, (__int64)v56, v22, v8, v9);
    if ( !v63 )
      goto LABEL_3;
  }
  v26 = 3;
  v61 = *(_QWORD *)(v18 + 24);
  v27 = 0;
  sub_15897D0(&v67, *(_DWORD *)(v19 + 8) >> 8, a4 == v61);
  v28 = *(_DWORD *)(v10 + 20);
  v59 = ((v28 & 0xFFFFFFFu) >> 1) - 1;
  if ( (v28 & 0xFFFFFFFu) >> 1 == 1 )
  {
LABEL_127:
    v38 = v68;
    v68 = 0;
    LODWORD(v77) = v38;
    v76 = v67;
    v39 = v70;
    v70 = 0;
    LODWORD(v79) = v39;
    v78 = v69;
    sub_13EA060(v81.m128i_i32, (__int64 *)&v76);
    sub_13E8810(&v71, (unsigned int *)&v81);
    if ( v81.m128i_i32[0] == 3 )
    {
      if ( v84 > 0x40 && v83 )
        j_j___libc_free_0_0(v83);
      if ( (unsigned int)v82 > 0x40 && v81.m128i_i64[1] )
        j_j___libc_free_0_0(v81.m128i_i64[1]);
    }
    if ( (unsigned int)v79 > 0x40 && v78 )
      j_j___libc_free_0_0(v78);
    if ( (unsigned int)v77 > 0x40 && v76 )
      j_j___libc_free_0_0(v76);
    v40 = 1;
    goto LABEL_135;
  }
  while ( 2 )
  {
    if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
      v29 = *(_QWORD *)(v10 - 8);
    else
      v29 = v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF);
    v30 = *(_QWORD *)(v29 + 24LL * (v26 - 1));
    v31 = *(_DWORD *)(v30 + 32);
    v66 = v31;
    if ( v31 <= 0x40 )
    {
      v32 = *(_QWORD *)(v30 + 24);
      v81.m128i_i32[2] = v31;
      v65 = v32;
      goto LABEL_56;
    }
    sub_16A4FD0(&v65, v30 + 24);
    v81.m128i_i32[2] = v66;
    if ( v66 <= 0x40 )
LABEL_56:
      v81.m128i_i64[0] = v65;
    else
      sub_16A4FD0(&v81, &v65);
    sub_1589870(&v76, &v81);
    if ( v81.m128i_i32[2] > 0x40u && v81.m128i_i64[0] )
      j_j___libc_free_0_0(v81.m128i_i64[0]);
    if ( !v63 )
      goto LABEL_61;
    v35 = sub_157EB90(a4);
    v36 = sub_1632FA0(v35);
    sub_13EA5B0(v81.m128i_i32, (__int64)a2, v56, (__int64)&v65, v36);
    if ( v81.m128i_i32[0] != 4 )
    {
      if ( (unsigned int)v77 <= 0x40 && (unsigned int)v82 <= 0x40 )
      {
        LODWORD(v77) = v82;
        v76 = v81.m128i_i64[1] & (0xFFFFFFFFFFFFFFFFLL >> -(char)v82);
      }
      else
      {
        sub_16A51C0(&v76, &v81.m128i_u64[1]);
      }
      if ( (unsigned int)v79 <= 0x40 && v84 <= 0x40 )
      {
        LODWORD(v79) = v84;
        v78 = v83 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v84);
        if ( v81.m128i_i32[0] != 3 )
          goto LABEL_61;
      }
      else
      {
        sub_16A51C0(&v78, &v83);
        if ( v81.m128i_i32[0] != 3 )
          goto LABEL_61;
        if ( v84 > 0x40 && v83 )
          j_j___libc_free_0_0(v83);
      }
      if ( (unsigned int)v82 > 0x40 && v81.m128i_i64[1] )
        j_j___libc_free_0_0(v81.m128i_i64[1]);
LABEL_61:
      v33 = 24;
      if ( a4 == v61 )
      {
        if ( (_DWORD)v27 != -2 )
          v33 = 24LL * v26;
        if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
          v37 = *(_QWORD *)(v10 - 8);
        else
          v37 = v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF);
        if ( *(_QWORD *)(v37 + v33) == a4 || a2 != v56 )
        {
LABEL_66:
          if ( (unsigned int)v79 > 0x40 && v78 )
            j_j___libc_free_0_0(v78);
          if ( (unsigned int)v77 > 0x40 && v76 )
            j_j___libc_free_0_0(v76);
          if ( v66 > 0x40 && v65 )
            j_j___libc_free_0_0(v65);
          ++v27;
          v26 += 2;
          if ( v59 == v27 )
            goto LABEL_127;
          continue;
        }
        sub_1590FF0(&v81, &v67, &v76);
        if ( v68 <= 0x40 )
          goto LABEL_81;
      }
      else
      {
        if ( (_DWORD)v27 != -2 )
          v33 = 24LL * v26;
        if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
        {
          if ( a4 != *(_QWORD *)(*(_QWORD *)(v10 - 8) + v33) )
            goto LABEL_66;
        }
        else if ( a4 != *(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF) + v33) )
        {
          goto LABEL_66;
        }
        sub_158C3A0(&v81, &v67, &v76);
        if ( v68 <= 0x40 )
        {
LABEL_81:
          v67 = v81.m128i_i64[0];
          v34 = v81.m128i_i32[2];
          v81.m128i_i32[2] = 0;
          v68 = v34;
          if ( v70 > 0x40 && v69 )
          {
            j_j___libc_free_0_0(v69);
            v69 = v82;
            v70 = v83;
            if ( v81.m128i_i32[2] > 0x40u && v81.m128i_i64[0] )
              j_j___libc_free_0_0(v81.m128i_i64[0]);
          }
          else
          {
            v69 = v82;
            v70 = v83;
          }
          goto LABEL_66;
        }
      }
      if ( v67 )
        j_j___libc_free_0_0(v67);
      goto LABEL_81;
    }
    break;
  }
  if ( (unsigned int)v79 > 0x40 && v78 )
    j_j___libc_free_0_0(v78);
  if ( (unsigned int)v77 > 0x40 && v76 )
    j_j___libc_free_0_0(v76);
  if ( v66 > 0x40 && v65 )
    j_j___libc_free_0_0(v65);
  v40 = 0;
LABEL_135:
  if ( v70 > 0x40 && v69 )
    j_j___libc_free_0_0(v69);
  if ( v68 > 0x40 && v67 )
    j_j___libc_free_0_0(v67);
  if ( !v40 )
  {
LABEL_3:
    v81.m128i_i32[0] = 4;
    sub_13E8810(&v71, (unsigned int *)&v81);
    if ( v81.m128i_i32[0] == 3 )
    {
      if ( v84 > 0x40 && v83 )
        j_j___libc_free_0_0(v83);
      if ( (unsigned int)v82 > 0x40 && v81.m128i_i64[1] )
        j_j___libc_free_0_0(v81.m128i_i64[1]);
    }
  }
  v11 = v71;
LABEL_5:
  if ( v11 == 3 )
  {
    v81.m128i_i32[2] = v73;
    if ( v73 > 0x40 )
      sub_16A4FD0(&v81, &v72);
    else
      v81.m128i_i64[0] = v72;
    sub_16A7490(&v81, 1);
    v23 = v81.m128i_u32[2];
    v24 = v81.m128i_i64[0];
    v81.m128i_i32[2] = 0;
    LODWORD(v77) = v23;
    v76 = v81.m128i_i64[0];
    if ( v75 <= 0x40 )
      v25 = v74 == v81.m128i_i64[0];
    else
      v25 = sub_16A5220(&v74, &v76);
    if ( v23 > 0x40 )
    {
      if ( v24 )
      {
        j_j___libc_free_0_0(v24);
        if ( v81.m128i_i32[2] > 0x40u )
        {
          if ( v81.m128i_i64[0] )
            j_j___libc_free_0_0(v81.m128i_i64[0]);
        }
      }
    }
    if ( !v25 )
    {
      v11 = v71;
      goto LABEL_6;
    }
    goto LABEL_32;
  }
LABEL_6:
  if ( v11 == 1 )
    goto LABEL_32;
  if ( a2[16] > 0x10u )
  {
    v12 = sub_13E8A40(a1, (__int64)a2, a3);
    if ( !(_BYTE)v12 )
    {
      v81.m128i_i64[1] = (__int64)a2;
      v81.m128i_i64[0] = a3;
      if ( (unsigned __int8)sub_13ED650(a1, &v81) )
        goto LABEL_12;
LABEL_32:
      v12 = 1;
      sub_13E8810(a5, (unsigned int *)&v71);
      if ( v71 != 3 )
        return v12;
      goto LABEL_33;
    }
  }
  sub_13E9630((int *)&v76, a1, (__int64)a2, a3);
  v13 = sub_157EBA0(a3);
  sub_13EE9C0(a1, (__int64)a2, (int *)&v76, v13);
  sub_13EE9C0(a1, (__int64)a2, (int *)&v76, a6);
  sub_13EA210(v81.m128i_i32, (__int64)&v71, (__int64)&v76);
  sub_13E8810(a5, (unsigned int *)&v81);
  if ( v81.m128i_i32[0] == 3 )
  {
    if ( v84 > 0x40 && v83 )
      j_j___libc_free_0_0(v83);
    if ( (unsigned int)v82 > 0x40 && v81.m128i_i64[1] )
      j_j___libc_free_0_0(v81.m128i_i64[1]);
  }
  if ( (_DWORD)v76 == 3 )
  {
    if ( v80 > 0x40 && v79 )
      j_j___libc_free_0_0(v79);
    if ( (unsigned int)v78 > 0x40 && v77 )
      j_j___libc_free_0_0(v77);
  }
  v12 = 1;
LABEL_12:
  if ( v71 == 3 )
  {
LABEL_33:
    if ( v75 > 0x40 && v74 )
      j_j___libc_free_0_0(v74);
    if ( v73 > 0x40 && v72 )
      j_j___libc_free_0_0(v72);
  }
  return v12;
}
