// Function: sub_1CF5980
// Address: 0x1cf5980
//
unsigned __int64 __fastcall sub_1CF5980(
        __int64 a1,
        __int64 *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // rdx
  __int64 v11; // rsi
  unsigned __int64 v12; // rdx
  char v13; // al
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // rax
  char v17; // cl
  __int64 v18; // rdx
  __int64 v20; // rax
  __int64 v21; // rcx
  unsigned __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // r12
  int v26; // r9d
  unsigned __int8 *v27; // rsi
  __int64 *v28; // r15
  __int64 v29; // r8
  __int64 v30; // rax
  __int64 *v31; // r15
  unsigned __int8 *v32; // rdx
  bool v33; // al
  _BYTE *v34; // rdx
  __int64 v35; // r15
  double v36; // xmm4_8
  double v37; // xmm5_8
  _QWORD *v38; // rdi
  __int64 *v39; // r13
  unsigned __int8 *v40; // rsi
  __int64 v41; // rax
  __int64 v42; // rax
  unsigned __int8 *v43; // rsi
  __int64 v44; // rdi
  __int64 *v45; // rbx
  __int64 v46; // rax
  __int64 v47; // rcx
  __int64 v48; // rsi
  unsigned __int8 *v49; // rsi
  __int64 v50; // [rsp+0h] [rbp-150h]
  unsigned __int8 *v51; // [rsp+18h] [rbp-138h] BYREF
  __int64 v52[2]; // [rsp+20h] [rbp-130h] BYREF
  __int16 v53; // [rsp+30h] [rbp-120h]
  unsigned __int8 *v54[2]; // [rsp+40h] [rbp-110h] BYREF
  __int64 v55; // [rsp+50h] [rbp-100h]
  __int64 v56; // [rsp+58h] [rbp-F8h]
  __int64 v57; // [rsp+60h] [rbp-F0h]
  int v58; // [rsp+68h] [rbp-E8h]
  __int64 v59; // [rsp+70h] [rbp-E0h]
  __int64 v60; // [rsp+78h] [rbp-D8h]
  unsigned __int8 *v61; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v62; // [rsp+98h] [rbp-B8h]
  _QWORD v63[3]; // [rsp+A0h] [rbp-B0h] BYREF
  int v64; // [rsp+B8h] [rbp-98h]
  __int64 v65; // [rsp+C0h] [rbp-90h]
  __int64 v66; // [rsp+C8h] [rbp-88h]

  v10 = *a2;
  v11 = *(_QWORD *)(a1 + 40);
  v12 = v10 & 0xFFFFFFFFFFFFFFF8LL;
  v13 = *(_BYTE *)(v12 + 16);
  if ( v13 == 54 )
    goto LABEL_2;
  if ( v13 != 56 )
  {
    if ( v13 != 71 )
      goto LABEL_34;
LABEL_2:
    v14 = *(_QWORD *)(v12 - 24);
    if ( v14 )
      goto LABEL_3;
LABEL_31:
    if ( v11 )
      goto LABEL_5;
    return v12;
  }
  v14 = *(_QWORD *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF));
  if ( !v14 )
    goto LABEL_31;
LABEL_3:
  if ( *(_BYTE *)(v14 + 16) > 0x17u )
  {
    if ( v11 != v14 )
      goto LABEL_5;
    v14 = v12;
LABEL_21:
    if ( v13 == 54 )
    {
      v20 = *(_QWORD *)(a1 + 96);
      if ( *(_QWORD *)(v14 - 24) )
      {
        v21 = *(_QWORD *)(v14 - 16);
        v22 = *(_QWORD *)(v14 - 8) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v22 = v21;
        if ( v21 )
          *(_QWORD *)(v21 + 16) = *(_QWORD *)(v21 + 16) & 3LL | v22;
      }
      *(_QWORD *)(v14 - 24) = v20;
      if ( !v20 )
        return 0;
      v23 = *(_QWORD *)(v20 + 8);
      *(_QWORD *)(v14 - 16) = v23;
      if ( v23 )
        *(_QWORD *)(v23 + 16) = (v14 - 16) | *(_QWORD *)(v23 + 16) & 3LL;
      v24 = *(_QWORD *)(v14 - 8);
      v25 = v14 - 24;
      *(_QWORD *)(v25 + 16) = (v20 + 8) | v24 & 3;
      *(_QWORD *)(v20 + 8) = v25;
      return 0;
    }
    if ( v13 != 71 )
    {
      v54[0] = 0;
      v56 = sub_16498A0(v14);
      v57 = 0;
      v58 = 0;
      v59 = 0;
      v60 = 0;
      v54[1] = *(unsigned __int8 **)(v14 + 40);
      v55 = v14 + 24;
      v27 = *(unsigned __int8 **)(v14 + 48);
      v61 = v27;
      if ( v27 )
      {
        sub_1623A60((__int64)&v61, (__int64)v27, 2);
        v54[0] = v61;
        if ( v61 )
          sub_1623210((__int64)&v61, v61, (__int64)v54);
      }
      v62 = 0x1000000000LL;
      v61 = (unsigned __int8 *)v63;
      v28 = (__int64 *)(v14 + 24 * (1LL - (*(_DWORD *)(v14 + 20) & 0xFFFFFFF)));
      if ( v28 != (__int64 *)v14 )
      {
        v29 = *v28;
        v30 = 0;
        v31 = v28 + 3;
        v32 = (unsigned __int8 *)v63;
        while ( 1 )
        {
          *(_QWORD *)&v32[8 * v30] = v29;
          v30 = (unsigned int)(v62 + 1);
          LODWORD(v62) = v62 + 1;
          if ( (__int64 *)v14 == v31 )
            break;
          v29 = *v31;
          if ( HIDWORD(v62) <= (unsigned int)v30 )
          {
            v50 = *v31;
            sub_16CD150((__int64)&v61, v63, 0, 8, v29, v26);
            v30 = (unsigned int)v62;
            v29 = v50;
          }
          v32 = v61;
          v31 += 3;
        }
      }
      v33 = sub_15FA300(v14);
      v53 = 257;
      v34 = *(_BYTE **)(a1 + 96);
      if ( v33 )
        v35 = sub_128B460((__int64 *)v54, 0, v34, (__int64 **)v61, (unsigned int)v62, (__int64)v52);
      else
        v35 = sub_1BBF860((__int64 *)v54, 0, v34, (__int64 **)v61, (unsigned int)v62, v52);
      if ( v61 != (unsigned __int8 *)v63 )
        _libc_free((unsigned __int64)v61);
      if ( v54[0] )
        sub_161E7C0((__int64)v54, (__int64)v54[0]);
LABEL_52:
      sub_164D160(v14, v35, a3, a4, a5, a6, v36, v37, a9, a10);
      v38 = (_QWORD *)v14;
      v14 = 0;
      sub_15F20C0(v38);
      return v14;
    }
    v39 = **(__int64 ***)(*(_QWORD *)v14 + 16LL);
    v61 = 0;
    v63[1] = sub_16498A0(v14);
    v63[2] = 0;
    v64 = 0;
    v65 = 0;
    v66 = 0;
    v62 = *(_QWORD *)(v14 + 40);
    v63[0] = v14 + 24;
    v40 = *(unsigned __int8 **)(v14 + 48);
    v54[0] = v40;
    if ( v40 )
    {
      sub_1623A60((__int64)v54, (__int64)v40, 2);
      if ( v61 )
        sub_161E7C0((__int64)&v61, (__int64)v61);
      v61 = v54[0];
      if ( v54[0] )
        sub_1623210((__int64)v54, v54[0], (__int64)&v61);
    }
    v53 = 257;
    v41 = sub_1646BA0(v39, 0);
    v35 = *(_QWORD *)(a1 + 96);
    if ( v41 != *(_QWORD *)v35 )
    {
      if ( *(_BYTE *)(v35 + 16) <= 0x10u )
      {
        v42 = sub_15A4AD0(*(__int64 ****)(a1 + 96), v41);
        v43 = v61;
        v35 = v42;
        goto LABEL_62;
      }
      v44 = *(_QWORD *)(a1 + 96);
      LOWORD(v55) = 257;
      v35 = sub_15FDF90(v44, v41, (__int64)v54, 0);
      if ( v62 )
      {
        v45 = (__int64 *)v63[0];
        sub_157E9D0(v62 + 40, v35);
        v46 = *(_QWORD *)(v35 + 24);
        v47 = *v45;
        *(_QWORD *)(v35 + 32) = v45;
        v47 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v35 + 24) = v47 | v46 & 7;
        *(_QWORD *)(v47 + 8) = v35 + 24;
        *v45 = *v45 & 7 | (v35 + 24);
      }
      sub_164B780(v35, v52);
      if ( !v61 )
        goto LABEL_52;
      v51 = v61;
      sub_1623A60((__int64)&v51, (__int64)v61, 2);
      v48 = *(_QWORD *)(v35 + 48);
      if ( v48 )
        sub_161E7C0(v35 + 48, v48);
      v49 = v51;
      *(_QWORD *)(v35 + 48) = v51;
      if ( v49 )
        sub_1623210((__int64)&v51, v49, v35 + 48);
    }
    v43 = v61;
LABEL_62:
    if ( v43 )
      sub_161E7C0((__int64)&v61, (__int64)v43);
    goto LABEL_52;
  }
LABEL_34:
  if ( !v11 )
    return v12;
  v14 = 0;
LABEL_5:
  v15 = *(_QWORD *)(v14 + 8);
  if ( v15 )
  {
    while ( 1 )
    {
      v16 = *(_QWORD *)(v15 + 8);
      if ( v16 )
        return v14;
      v17 = *(_BYTE *)(v12 + 16);
      if ( v17 == 54 )
        break;
      if ( v17 == 56 )
      {
        v18 = *(_QWORD *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF));
        if ( !v18 )
          goto LABEL_11;
        goto LABEL_9;
      }
      if ( v17 == 71 )
        break;
      v12 = v14;
      if ( !v11 )
      {
LABEL_19:
        if ( !v11 )
          return v14;
        v13 = *(_BYTE *)(v14 + 16);
        goto LABEL_21;
      }
LABEL_12:
      v14 = v16;
      v15 = *(_QWORD *)(v16 + 8);
      if ( !v15 )
        return v14;
    }
    v18 = *(_QWORD *)(v12 - 24);
    if ( !v18 )
      goto LABEL_11;
LABEL_9:
    if ( *(_BYTE *)(v18 + 16) >= 0x18u )
      v16 = v18;
LABEL_11:
    v12 = v14;
    if ( v11 == v16 )
      goto LABEL_19;
    goto LABEL_12;
  }
  return v14;
}
