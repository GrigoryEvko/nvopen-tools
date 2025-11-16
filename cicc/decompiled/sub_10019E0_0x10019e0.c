// Function: sub_10019E0
// Address: 0x10019e0
//
__int64 __fastcall sub_10019E0(unsigned int a1, __int64 a2, unsigned __int8 *a3, const __m128i *a4)
{
  __int64 v5; // r13
  bool v6; // r8
  __int64 result; // rax
  unsigned int v8; // eax
  unsigned int v9; // r15d
  unsigned __int8 *v10; // rbx
  unsigned __int8 *v11; // r12
  unsigned __int8 v12; // al
  unsigned __int8 v13; // al
  __int64 v14; // rdi
  int v15; // r14d
  unsigned int v16; // r14d
  int v17; // r14d
  _QWORD *v18; // r15
  __int64 v19; // rax
  int v20; // eax
  unsigned __int64 v21; // rdx
  bool v22; // r14
  unsigned __int8 v23; // r12
  __int64 v24; // rax
  _QWORD *v25; // r15
  _QWORD *v26; // r15
  unsigned __int8 v27; // al
  _QWORD **v28; // rdx
  __int64 v29; // r12
  int v30; // ecx
  int v31; // eax
  __int64 *v32; // rax
  __int64 v33; // rdi
  __int64 v34; // rsi
  __int64 (__fastcall ***v35)(); // rsi
  unsigned __int8 v36; // r12
  __int64 v37; // rax
  unsigned __int8 v38; // al
  _QWORD **v39; // rdx
  __int64 v40; // r12
  int v41; // ecx
  int v42; // eax
  __int64 *v43; // rax
  __int64 v44; // rdi
  char v45; // al
  int v46; // edx
  unsigned __int8 v47; // al
  unsigned __int64 v48; // rdx
  unsigned int v49; // r14d
  _QWORD *v50; // r15
  unsigned int v51; // [rsp+8h] [rbp-148h]
  int v52; // [rsp+14h] [rbp-13Ch]
  __int64 *v54; // [rsp+20h] [rbp-130h]
  __int64 v55; // [rsp+38h] [rbp-118h]
  __int64 v56; // [rsp+38h] [rbp-118h]
  unsigned __int64 v57; // [rsp+38h] [rbp-118h]
  __int64 v58; // [rsp+38h] [rbp-118h]
  _QWORD *v59; // [rsp+38h] [rbp-118h]
  _QWORD *v60; // [rsp+38h] [rbp-118h]
  __int64 v61; // [rsp+38h] [rbp-118h]
  __int64 v62; // [rsp+38h] [rbp-118h]
  __int64 v63; // [rsp+38h] [rbp-118h]
  unsigned __int64 v64; // [rsp+40h] [rbp-110h] BYREF
  unsigned __int64 v65; // [rsp+48h] [rbp-108h] BYREF
  _QWORD *v66; // [rsp+50h] [rbp-100h] BYREF
  unsigned int v67; // [rsp+58h] [rbp-F8h]
  __int64 v68; // [rsp+60h] [rbp-F0h] BYREF
  unsigned int v69; // [rsp+68h] [rbp-E8h]
  __int64 (__fastcall **v70)(); // [rsp+70h] [rbp-E0h] BYREF
  int v71; // [rsp+78h] [rbp-D8h]
  unsigned __int64 v72; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v73; // [rsp+88h] [rbp-C8h]
  _BYTE v74[64]; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v75; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v76; // [rsp+D8h] [rbp-78h]
  _BYTE v77[112]; // [rsp+E0h] [rbp-70h] BYREF

  v5 = a4->m128i_i64[0];
  v54 = (__int64 *)a4->m128i_i64[1];
  v6 = sub_B532B0(a1);
  result = 0;
  if ( v6 )
    return result;
  v52 = sub_B52E90(a1);
  v8 = sub_AE43F0(v5, *(_QWORD *)(a2 + 8));
  v67 = v8;
  v9 = v8;
  if ( v8 > 0x40 )
  {
    sub_C43690((__int64)&v66, 0, 0);
    v69 = v9;
    sub_C43690((__int64)&v68, 0, 0);
  }
  else
  {
    v69 = v8;
    v66 = 0;
    v68 = 0;
  }
  v10 = sub_BD45C0((unsigned __int8 *)a2, v5, (__int64)&v66, (unsigned int)(v52 - 32) <= 1, 0, 0, 0, 0);
  v11 = sub_BD45C0(a3, v5, (__int64)&v68, (unsigned int)(v52 - 32) <= 1, 0, 0, 0, 0);
  if ( v10 == v11 )
  {
    v38 = sub_B532C0((__int64)&v66, &v68, v52);
    v39 = (_QWORD **)*((_QWORD *)v10 + 1);
    v40 = v38;
    v41 = *((unsigned __int8 *)v39 + 8);
    if ( (unsigned int)(v41 - 17) > 1 )
    {
      v44 = sub_BCB2A0(*v39);
    }
    else
    {
      v42 = *((_DWORD *)v39 + 8);
      BYTE4(v75) = (_BYTE)v41 == 18;
      LODWORD(v75) = v42;
      v43 = (__int64 *)sub_BCB2A0(*v39);
      v44 = sub_BCE1B0(v43, v75);
    }
    result = sub_AD64C0(v44, v40, 0);
    goto LABEL_6;
  }
  result = 0;
  if ( (unsigned int)(v52 - 32) <= 1 )
  {
    if ( *v10 == 22 && (unsigned __int8)sub_B2D680((__int64)v10) )
    {
      v12 = *v11;
      if ( *v11 <= 0x1Cu )
      {
        if ( v12 != 3 && (v12 != 22 || !(unsigned __int8)sub_B2D680((__int64)v11)) )
          goto LABEL_41;
      }
      else if ( v12 != 60 )
      {
        goto LABEL_41;
      }
      goto LABEL_75;
    }
    if ( *v11 == 22 && (unsigned __int8)sub_B2D680((__int64)v11) )
    {
      v13 = *v10;
      if ( *v10 <= 0x1Cu )
      {
        v51 = 2;
        if ( v13 == 3 )
          goto LABEL_71;
        if ( v13 != 22 || !(unsigned __int8)sub_B2D680((__int64)v10) )
          goto LABEL_41;
LABEL_75:
        HIWORD(v46) = HIWORD(v51);
        LOWORD(v46) = 2;
        v51 = v46 & 0xFF00FFFF;
        if ( *v10 <= 0x1Cu )
        {
          if ( *v10 != 22 )
            goto LABEL_71;
          v14 = *((_QWORD *)v10 + 3);
LABEL_25:
          if ( v14 )
          {
            v15 = (unsigned __int8)sub_B2F070(v14, 0);
LABEL_27:
            v16 = v51 & 0xFF00FFFF | (v15 << 16);
            if ( !(unsigned __int8)sub_D62CA0((__int64)v10, &v64, v5, (__int64)v54, v16, 0)
              || !v64
              || !(unsigned __int8)sub_D62CA0((__int64)v11, &v65, v5, (__int64)v54, v16, 0)
              || !v65 )
            {
              goto LABEL_41;
            }
            LODWORD(v76) = v67;
            if ( v67 > 0x40 )
              sub_C43780((__int64)&v75, (const void **)&v66);
            else
              v75 = (__int64)v66;
            sub_C46B40((__int64)&v75, &v68);
            v17 = v76;
            v18 = (_QWORD *)v75;
            v71 = v76;
            v19 = 1LL << ((unsigned __int8)v76 - 1);
            v70 = (__int64 (__fastcall **)())v75;
            if ( (unsigned int)v76 <= 0x40 )
            {
              if ( (v19 & v75) == 0 )
              {
                v21 = v64;
                goto LABEL_37;
              }
            }
            else if ( (*(_QWORD *)(v75 + 8LL * ((unsigned int)(v76 - 1) >> 6)) & v19) == 0 )
            {
              v57 = v64;
              v20 = sub_C444A0((__int64)&v70);
              v21 = v57;
              if ( (unsigned int)(v17 - v20) <= 0x40 )
              {
                v18 = (_QWORD *)*v18;
LABEL_37:
                v22 = v21 > (unsigned __int64)v18;
                goto LABEL_38;
              }
LABEL_90:
              sub_969240((__int64 *)&v70);
              goto LABEL_41;
            }
            sub_9865C0((__int64)&v72, (__int64)&v70);
            if ( (unsigned int)v73 > 0x40 )
            {
              sub_C43D10((__int64)&v72);
            }
            else
            {
              v48 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v73;
              if ( !(_DWORD)v73 )
                v48 = 0;
              v72 = v48 & ~v72;
            }
            sub_C46250((__int64)&v72);
            v49 = v73;
            v50 = (_QWORD *)v72;
            LODWORD(v73) = 0;
            LODWORD(v76) = v49;
            v75 = v72;
            if ( v49 > 0x40 )
            {
              if ( v49 - (unsigned int)sub_C444A0((__int64)&v75) > 0x40 )
              {
                sub_969240(&v75);
                sub_969240((__int64 *)&v72);
                goto LABEL_90;
              }
              v50 = (_QWORD *)*v50;
            }
            v22 = v65 > (unsigned __int64)v50;
            sub_969240(&v75);
            sub_969240((__int64 *)&v72);
LABEL_38:
            if ( v22 )
            {
              v23 = sub_B535D0(v52);
              v24 = sub_1001990(*((_QWORD ***)v10 + 1));
              v58 = sub_AD64C0(v24, v23 ^ 1u, 0);
              sub_969240((__int64 *)&v70);
              result = v58;
              goto LABEL_6;
            }
            goto LABEL_90;
          }
LABEL_71:
          v15 = 1;
          goto LABEL_27;
        }
LABEL_24:
        v14 = sub_B43CB0((__int64)v10);
        goto LABEL_25;
      }
      if ( v13 == 60 )
        goto LABEL_23;
    }
    else
    {
      if ( *v10 != 60 )
        goto LABEL_41;
      v47 = *v11;
      if ( *v11 > 0x1Cu )
      {
        if ( v47 != 60 )
          goto LABEL_41;
        goto LABEL_23;
      }
      if ( v47 == 3 )
      {
LABEL_23:
        v51 = 2;
        goto LABEL_24;
      }
    }
LABEL_41:
    v73 = 0x800000000LL;
    v76 = 0x800000000LL;
    v72 = (unsigned __int64)v74;
    v75 = (__int64)v77;
    sub_98B4D0((__int64)v10, (__int64)&v72, 0, 6u);
    sub_98B4D0((__int64)v11, (__int64)&v75, 0, 6u);
    v25 = (_QWORD *)(v72 + 8LL * (unsigned int)v73);
    if ( v25 == sub_FFEA90((_QWORD *)v72, (__int64)v25, (unsigned __int8 (__fastcall *)(_QWORD))sub_CF6FD0)
      && (v59 = (_QWORD *)(v75 + 8LL * (unsigned int)v76),
          v59 == sub_FFEA90((_QWORD *)v75, (__int64)v59, (unsigned __int8 (__fastcall *)(_QWORD))sub_FFECD0))
      || (v60 = (_QWORD *)(v75 + 8LL * (unsigned int)v76),
          v60 == sub_FFEA90((_QWORD *)v75, (__int64)v60, (unsigned __int8 (__fastcall *)(_QWORD))sub_CF6FD0))
      && (v26 = (_QWORD *)(v72 + 8LL * (unsigned int)v73),
          v26 == sub_FFEA90((_QWORD *)v72, (__int64)v26, (unsigned __int8 (__fastcall *)(_QWORD))sub_FFECD0)) )
    {
      v27 = sub_B535D0(v52);
      v28 = (_QWORD **)*((_QWORD *)v10 + 1);
      v29 = v27 ^ 1u;
      v30 = *((unsigned __int8 *)v28 + 8);
      if ( (unsigned int)(v30 - 17) > 1 )
      {
        v33 = sub_BCB2A0(*v28);
      }
      else
      {
        v31 = *((_DWORD *)v28 + 8);
        BYTE4(v70) = (_BYTE)v30 == 18;
        LODWORD(v70) = v31;
        v32 = (__int64 *)sub_BCB2A0(*v28);
        v33 = sub_BCE1B0(v32, (__int64)v70);
      }
      v34 = v29;
      result = sub_AD64C0(v33, v29, 0);
      goto LABEL_53;
    }
    if ( (unsigned __int8)sub_D5CC50(v10, v54) && (unsigned __int8)sub_9B6260((__int64)v11, a4, 0) )
    {
      v11 = v10;
    }
    else
    {
      v35 = (__int64 (__fastcall ***)())v54;
      if ( !(unsigned __int8)sub_D5CC50(v11, v54) )
        goto LABEL_58;
      v35 = (__int64 (__fastcall ***)())a4;
      v45 = sub_9B6260((__int64)v10, a4, 0);
      if ( !v11 || !v45 )
        goto LABEL_58;
    }
    v35 = &v70;
    v70 = off_49E5658;
    LOBYTE(v71) = 0;
    sub_D13D60((__int64)v11, (__int64)&v70, 0);
    if ( !(_BYTE)v71 )
    {
      v36 = sub_B53600(v52);
      v37 = sub_1001990(*((_QWORD ***)v10 + 1));
      v34 = v36;
      v61 = sub_AD64C0(v37, v36, 0);
      v70 = off_49E5658;
      nullsub_185();
      result = v61;
LABEL_53:
      if ( (_BYTE *)v75 != v77 )
      {
        v62 = result;
        _libc_free(v75, v34);
        result = v62;
      }
      if ( (_BYTE *)v72 != v74 )
      {
        v63 = result;
        _libc_free(v72, v34);
        result = v63;
      }
      goto LABEL_6;
    }
    v70 = off_49E5658;
    nullsub_185();
LABEL_58:
    if ( (_BYTE *)v75 != v77 )
      _libc_free(v75, v35);
    if ( (_BYTE *)v72 != v74 )
      _libc_free(v72, v35);
    result = 0;
  }
LABEL_6:
  if ( v69 > 0x40 && v68 )
  {
    v55 = result;
    j_j___libc_free_0_0(v68);
    result = v55;
  }
  if ( v67 > 0x40 )
  {
    if ( v66 )
    {
      v56 = result;
      j_j___libc_free_0_0(v66);
      return v56;
    }
  }
  return result;
}
