// Function: sub_111CED0
// Address: 0x111ced0
//
_QWORD *__fastcall sub_111CED0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int16 v6; // r12
  char v7; // bl
  int v8; // r12d
  __int16 v9; // bx
  unsigned int v10; // edx
  unsigned int v11; // r15d
  __int64 v12; // rdx
  _QWORD *v13; // r15
  __int64 v14; // r14
  __int64 v15; // rdx
  __int64 v17; // rdx
  _BYTE *v18; // rcx
  __int64 *v19; // rsi
  _BYTE *v20; // rax
  unsigned int v21; // r14d
  _QWORD *v22; // rax
  unsigned int v23; // r15d
  __int64 *v24; // r14
  __int64 v25; // rax
  const void **v26; // r10
  __int64 v27; // r13
  unsigned int v28; // eax
  unsigned __int64 v29; // rcx
  __int64 v30; // rbx
  __int64 v31; // rdx
  int v32; // ecx
  int v33; // eax
  _QWORD *v34; // rdi
  __int64 *v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rax
  int v38; // eax
  __int64 v39; // rsi
  __int64 v40; // r14
  __int64 v41; // rdx
  unsigned int v42; // esi
  __int64 v43; // rsi
  __int64 v44; // [rsp+0h] [rbp-E0h]
  _QWORD **v45; // [rsp+8h] [rbp-D8h]
  __int64 i; // [rsp+8h] [rbp-D8h]
  __int64 v47; // [rsp+10h] [rbp-D0h]
  int v48; // [rsp+18h] [rbp-C8h]
  __int64 v49; // [rsp+18h] [rbp-C8h]
  __int64 v50; // [rsp+18h] [rbp-C8h]
  const void **v51; // [rsp+18h] [rbp-C8h]
  __int64 v52; // [rsp+28h] [rbp-B8h] BYREF
  __int64 v53; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v54; // [rsp+38h] [rbp-A8h]
  __int64 v55; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v56; // [rsp+48h] [rbp-98h]
  __int64 v57; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v58; // [rsp+58h] [rbp-88h]
  __int16 v59; // [rsp+70h] [rbp-70h]
  __int64 v60; // [rsp+80h] [rbp-60h] BYREF
  __int64 *v61; // [rsp+88h] [rbp-58h]
  __int64 *v62; // [rsp+90h] [rbp-50h] BYREF
  char v63; // [rsp+98h] [rbp-48h]
  __int16 v64; // [rsp+A0h] [rbp-40h]

  v6 = *(_WORD *)(a2 + 2);
  v56 = 1;
  v55 = 0;
  v7 = v6;
  v8 = v6 & 0x3F;
  v9 = v7 & 0x3F;
  if ( v9 == 36 )
  {
    if ( *(_DWORD *)(a4 + 8) <= 0x40u )
    {
      v15 = *(_QWORD *)a4;
      v56 = *(_DWORD *)(a4 + 8);
      v55 = v15;
      goto LABEL_17;
    }
    sub_C43990((__int64)&v55, a4);
    v11 = v56;
  }
  else
  {
    if ( v8 != 34 )
      return 0;
    v10 = *(_DWORD *)(a4 + 8);
    if ( !v10 )
      return 0;
    if ( v10 <= 0x40 )
    {
      v37 = *(_QWORD *)a4;
      if ( *(_QWORD *)a4 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v10) )
        return 0;
      LODWORD(v61) = *(_DWORD *)(a4 + 8);
      v60 = v37;
    }
    else
    {
      v48 = *(_DWORD *)(a4 + 8);
      if ( v48 == (unsigned int)sub_C445E0(a4) )
        return 0;
      LODWORD(v61) = v48;
      sub_C43780((__int64)&v60, (const void **)a4);
    }
    sub_C46A40((__int64)&v60, 1);
    v11 = (unsigned int)v61;
    LODWORD(v61) = 0;
    v55 = v60;
    v56 = v11;
  }
  if ( v11 > 0x40 )
  {
    if ( (unsigned int)sub_C44630((__int64)&v55) != 1 )
    {
      v14 = v55;
      goto LABEL_62;
    }
    goto LABEL_10;
  }
LABEL_17:
  if ( !v55 || (v55 & (v55 - 1)) != 0 )
    return 0;
LABEL_10:
  v63 = 0;
  v62 = &v53;
  v12 = *(_QWORD *)(a3 + 16);
  v60 = (__int64)&v52;
  v61 = &v52;
  if ( !v12 )
    goto LABEL_11;
  v13 = *(_QWORD **)(v12 + 8);
  if ( v13 || *(_BYTE *)a3 != 59 )
    goto LABEL_11;
  v17 = *(_QWORD *)(a3 - 64);
  v18 = *(_BYTE **)(a3 - 32);
  if ( !v17 )
    goto LABEL_74;
  v52 = *(_QWORD *)(a3 - 64);
  v19 = &v52;
  if ( *v18 != 56 )
    goto LABEL_25;
  v43 = *((_QWORD *)v18 - 8);
  if ( v17 != v43 || !v43 )
  {
    v19 = &v52;
    goto LABEL_25;
  }
  if ( !(unsigned __int8)sub_991580((__int64)&v62, *((_QWORD *)v18 - 4)) )
  {
    v18 = *(_BYTE **)(a3 - 32);
LABEL_74:
    if ( !v18 )
      goto LABEL_11;
    v19 = (__int64 *)v60;
LABEL_25:
    *v19 = (__int64)v18;
    v20 = *(_BYTE **)(a3 - 64);
    if ( *v20 == 56 && *((_QWORD *)v20 - 8) == *v61 && (unsigned __int8)sub_991580((__int64)&v62, *((_QWORD *)v20 - 4)) )
      goto LABEL_28;
LABEL_11:
    v13 = 0;
    goto LABEL_12;
  }
LABEL_28:
  v21 = *(_DWORD *)(v53 + 8);
  if ( v21 > 0x40 )
  {
    v50 = v52;
    v45 = (_QWORD **)v53;
    v38 = sub_C444A0(v53);
    v47 = *(_QWORD *)(v50 + 8);
    if ( v21 - v38 > 0x40 )
      goto LABEL_31;
    v22 = (_QWORD *)**v45;
  }
  else
  {
    v22 = *(_QWORD **)v53;
    v47 = *(_QWORD *)(v52 + 8);
  }
  if ( v22 )
  {
LABEL_31:
    v14 = v55;
    v23 = v56 - 1;
    if ( v56 > 0x40 )
    {
      if ( (*(_QWORD *)(v55 + 8LL * (v23 >> 6)) & (1LL << v23)) == 0 || v23 != (unsigned int)sub_C44590((__int64)&v55) )
      {
LABEL_34:
        v24 = *(__int64 **)(a1 + 32);
        v59 = 257;
        v49 = sub_AD8D80(v47, (__int64)&v55);
        v44 = v52;
        v25 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v24[10] + 32LL))(
                v24[10],
                13,
                v52,
                v49,
                0,
                0);
        v26 = (const void **)&v55;
        v27 = v25;
        if ( !v25 )
        {
          v64 = 257;
          v27 = sub_B504D0(13, v44, v49, (__int64)&v60, 0, 0);
          (*(void (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64))(*(_QWORD *)v24[11] + 16LL))(
            v24[11],
            v27,
            &v57,
            v24[7],
            v24[8]);
          v26 = (const void **)&v55;
          v39 = *v24 + 16LL * *((unsigned int *)v24 + 2);
          v40 = *v24;
          for ( i = v39; i != v40; v26 = v51 )
          {
            v41 = *(_QWORD *)(v40 + 8);
            v42 = *(_DWORD *)v40;
            v40 += 16;
            v51 = v26;
            sub_B99FD0(v27, v42, v41);
          }
        }
        v28 = v56;
        if ( v9 == 36 )
        {
          v58 = v56;
          if ( v56 > 0x40 )
            sub_C43780((__int64)&v57, v26);
          else
            v57 = v55;
          sub_1110B10((__int64)&v57, 1u);
          goto LABEL_42;
        }
        LODWORD(v61) = v56;
        if ( v56 > 0x40 )
        {
          sub_C43780((__int64)&v60, v26);
          v28 = (unsigned int)v61;
          if ( (unsigned int)v61 > 0x40 )
          {
            sub_C47690(&v60, 1u);
LABEL_41:
            sub_C46F20((__int64)&v60, 1u);
            v58 = (unsigned int)v61;
            v57 = v60;
LABEL_42:
            v30 = sub_AD8D80(v47, (__int64)&v57);
            v64 = 257;
            v13 = sub_BD2C40(72, unk_3F10FD0);
            if ( v13 )
            {
              v31 = *(_QWORD *)(v27 + 8);
              v32 = *(unsigned __int8 *)(v31 + 8);
              if ( (unsigned int)(v32 - 17) > 1 )
              {
                v36 = sub_BCB2A0(*(_QWORD **)v31);
              }
              else
              {
                v33 = *(_DWORD *)(v31 + 32);
                v34 = *(_QWORD **)v31;
                BYTE4(v54) = (_BYTE)v32 == 18;
                LODWORD(v54) = v33;
                v35 = (__int64 *)sub_BCB2A0(v34);
                v36 = sub_BCE1B0(v35, v54);
              }
              sub_B523C0((__int64)v13, v36, 53, v8, v27, v30, (__int64)&v60, 0, 0, 0);
            }
            if ( v58 > 0x40 && v57 )
              j_j___libc_free_0_0(v57);
            goto LABEL_12;
          }
        }
        else
        {
          v60 = v55;
        }
        v29 = 0;
        if ( v28 >= 2 )
          v29 = (2 * v60) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v28);
        v60 = v29;
        goto LABEL_41;
      }
LABEL_62:
      v13 = 0;
      goto LABEL_14;
    }
    if ( v55 != 1LL << v23 )
      goto LABEL_34;
    return 0;
  }
LABEL_12:
  if ( v56 <= 0x40 )
    return v13;
  v14 = v55;
LABEL_14:
  if ( v14 )
    j_j___libc_free_0_0(v14);
  return v13;
}
