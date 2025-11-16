// Function: sub_10A0A60
// Address: 0x10a0a60
//
_BYTE *__fastcall sub_10A0A60(_QWORD *a1, __int64 a2)
{
  _BYTE *v2; // rax
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rdi
  _BYTE *v8; // rdi
  unsigned int v9; // eax
  _BYTE *v10; // rdx
  _BYTE *v11; // r12
  unsigned __int64 v13; // rdx
  unsigned int v14; // eax
  unsigned int v15; // ebx
  __int64 v16; // rax
  unsigned int **v17; // r12
  _BYTE *v18; // rax
  __int64 v19; // rax
  unsigned int v20; // ebx
  unsigned int **v21; // r13
  _BYTE *v22; // rax
  _BYTE *v23; // rax
  __int64 v24; // r13
  _BYTE *v25; // r14
  __int64 *v26; // rbx
  _BYTE *v27; // r14
  __int64 *v28; // rbx
  __int64 v29; // r13
  __int64 v30; // rbx
  __int64 v31; // r13
  __int64 v32; // rdx
  unsigned int v33; // esi
  __int64 v34; // r13
  __int64 v35; // rbx
  __int64 v36; // r13
  __int64 v37; // rdx
  unsigned int v38; // esi
  _BYTE *v40; // [rsp+28h] [rbp-128h]
  unsigned __int8 v41; // [rsp+37h] [rbp-119h] BYREF
  _BYTE *v42; // [rsp+38h] [rbp-118h] BYREF
  _BYTE *v43; // [rsp+40h] [rbp-110h] BYREF
  _BYTE *v44; // [rsp+48h] [rbp-108h] BYREF
  _BYTE *v45; // [rsp+50h] [rbp-100h] BYREF
  _BYTE *v46; // [rsp+58h] [rbp-F8h] BYREF
  const void *v47; // [rsp+60h] [rbp-F0h] BYREF
  unsigned int v48; // [rsp+68h] [rbp-E8h]
  const void *v49; // [rsp+70h] [rbp-E0h] BYREF
  unsigned int v50; // [rsp+78h] [rbp-D8h]
  _BYTE *v51; // [rsp+80h] [rbp-D0h] BYREF
  unsigned int v52; // [rsp+88h] [rbp-C8h]
  _BYTE *v53; // [rsp+90h] [rbp-C0h] BYREF
  unsigned int v54; // [rsp+98h] [rbp-B8h]
  const void *v55; // [rsp+A0h] [rbp-B0h] BYREF
  unsigned int v56; // [rsp+A8h] [rbp-A8h]
  const void *v57; // [rsp+B0h] [rbp-A0h] BYREF
  unsigned int v58; // [rsp+B8h] [rbp-98h]
  bool v59[32]; // [rsp+C0h] [rbp-90h] BYREF
  __int16 v60; // [rsp+E0h] [rbp-70h]
  unsigned __int64 v61; // [rsp+F0h] [rbp-60h] BYREF
  unsigned int v62; // [rsp+F8h] [rbp-58h]
  __int16 v63; // [rsp+110h] [rbp-40h]

  v2 = *(_BYTE **)(a2 - 64);
  v3 = *(_QWORD *)(a2 - 32);
  v48 = 1;
  v40 = v2;
  v47 = 0;
  v50 = 1;
  v49 = 0;
  if ( (!sub_109D820(v2, &v42, (__int64)&v47, &v41) || !(unsigned __int8)sub_109D670((_BYTE *)v3, &v43, (__int64)&v49))
    && (!sub_109D820((_BYTE *)v3, &v42, (__int64)&v47, &v41) || !(unsigned __int8)sub_109D670(v40, &v43, (__int64)&v49)) )
  {
    goto LABEL_4;
  }
  if ( v48 <= 0x40 )
  {
    if ( v47 != v49 )
      goto LABEL_4;
  }
  else if ( !sub_C43C50((__int64)&v47, &v49) )
  {
    goto LABEL_4;
  }
  v56 = 1;
  v55 = 0;
  if ( !sub_109D820(v43, &v51, (__int64)&v55, &v46) || v41 != (_BYTE)v46 )
  {
LABEL_43:
    if ( v56 > 0x40 && v55 )
      j_j___libc_free_0_0(v55);
LABEL_4:
    v52 = 1;
    v51 = 0;
    v54 = 1;
    v53 = 0;
    v4 = *((_QWORD *)v40 + 2);
    if ( !v4 || *(_QWORD *)(v4 + 8) || !(unsigned __int8)sub_109D670(v40, &v44, (__int64)&v51) )
    {
      v5 = *(_QWORD *)(a2 + 8);
      v44 = v40;
      v62 = sub_BCB060(v5);
      if ( v62 > 0x40 )
        sub_C43690((__int64)&v61, 1, 0);
      else
        v61 = 1;
      if ( v52 > 0x40 && v51 )
        j_j___libc_free_0_0(v51);
      v51 = (_BYTE *)v61;
      v52 = v62;
    }
    v6 = *(_QWORD *)(v3 + 16);
    if ( !v6 || *(_QWORD *)(v6 + 8) || !(unsigned __int8)sub_109D670((_BYTE *)v3, &v45, (__int64)&v53) )
    {
      v7 = *(_QWORD *)(a2 + 8);
      v45 = (_BYTE *)v3;
      v62 = sub_BCB060(v7);
      if ( v62 > 0x40 )
        sub_C43690((__int64)&v61, 1, 0);
      else
        v61 = 1;
      if ( v54 > 0x40 && v53 )
        j_j___libc_free_0_0(v53);
      v53 = (_BYTE *)v61;
      v54 = v62;
    }
    v8 = v44;
    if ( (unsigned __int8)(*v44 - 51) > 1u )
    {
      v8 = v45;
    }
    else
    {
      v9 = v52;
      v44 = v45;
      v10 = v51;
      v45 = v8;
      v51 = v53;
      v53 = v10;
      v52 = v54;
      v54 = v9;
    }
    v56 = 1;
    v55 = 0;
    if ( sub_109D820(v8, &v42, (__int64)&v47, &v41)
      && (unsigned __int8)sub_109DB70(v44, &v46, (__int64)&v55, v41)
      && v42 == v46 )
    {
      if ( v48 <= 0x40 )
      {
        if ( v47 == v55 )
        {
LABEL_50:
          sub_C472A0((__int64)&v61, (__int64)&v53, (__int64 *)&v47);
          if ( v62 > 0x40 )
          {
            sub_C43D10((__int64)&v61);
          }
          else
          {
            v13 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v62;
            if ( !v62 )
              v13 = 0;
            v61 = v13 & ~v61;
          }
          sub_C46250((__int64)&v61);
          sub_C45EE0((__int64)&v61, (__int64 *)&v51);
          v14 = v62;
          v62 = 0;
          v58 = v14;
          v57 = (const void *)v61;
          sub_969240((__int64 *)&v61);
          v15 = v58;
          if ( v58 <= 0x40 )
          {
            if ( v57 )
              goto LABEL_56;
          }
          else if ( v15 != (unsigned int)sub_C444A0((__int64)&v57) )
          {
LABEL_56:
            v16 = *((_QWORD *)v45 + 2);
            if ( !v16 || *(_QWORD *)(v16 + 8) )
              goto LABEL_57;
          }
          if ( sub_98EF80(v42, a1[8], a2, a1[10], 0) )
          {
            v17 = (unsigned int **)a1[4];
            v63 = 257;
            v18 = (_BYTE *)sub_AD8D80(*((_QWORD *)v42 + 1), (__int64)&v53);
            v19 = sub_A81850(v17, v42, v18, (__int64)&v61, 0, 0);
            v20 = v58;
            v11 = (_BYTE *)v19;
            if ( v58 <= 0x40 )
            {
              if ( !v57 )
                goto LABEL_58;
              goto LABEL_80;
            }
            if ( v20 != (unsigned int)sub_C444A0((__int64)&v57) )
            {
LABEL_80:
              v60 = 257;
              v21 = (unsigned int **)a1[4];
              v63 = 257;
              v22 = (_BYTE *)sub_AD8D80(*((_QWORD *)v42 + 1), (__int64)&v57);
              v23 = (_BYTE *)sub_A81850(v21, v44, v22, (__int64)v59, 0, 0);
              v11 = (_BYTE *)sub_929C50(v21, v23, v11, (__int64)&v61, 0, 0);
            }
LABEL_58:
            sub_969240((__int64 *)&v57);
LABEL_18:
            if ( v56 > 0x40 && v55 )
              j_j___libc_free_0_0(v55);
            if ( v54 > 0x40 && v53 )
              j_j___libc_free_0_0(v53);
            if ( v52 > 0x40 && v51 )
              j_j___libc_free_0_0(v51);
            goto LABEL_27;
          }
LABEL_57:
          v11 = 0;
          goto LABEL_58;
        }
      }
      else if ( sub_C43C50((__int64)&v47, &v55) )
      {
        goto LABEL_50;
      }
    }
    v11 = 0;
    goto LABEL_18;
  }
  v58 = 1;
  v57 = 0;
  if ( !(unsigned __int8)sub_109DB70(v51, &v53, (__int64)&v57, v41) || v42 != v53 )
  {
LABEL_74:
    sub_969240((__int64 *)&v57);
    goto LABEL_43;
  }
  if ( v48 <= 0x40 )
  {
    if ( v47 != v57 )
      goto LABEL_74;
  }
  else if ( !sub_C43C50((__int64)&v47, &v57) )
  {
    goto LABEL_74;
  }
  if ( v41 )
    sub_C4A7C0((__int64)&v61, (__int64)&v47, (__int64)&v55, v59);
  else
    sub_C49BE0((__int64)&v61, (__int64)&v47, (__int64)&v55, v59);
  sub_969240((__int64 *)&v61);
  if ( v59[0] )
    goto LABEL_74;
  sub_C472A0((__int64)&v61, (__int64)&v47, (__int64 *)&v55);
  v24 = sub_AD8D80(*((_QWORD *)v42 + 1), (__int64)&v61);
  sub_969240((__int64 *)&v61);
  if ( v41 )
  {
    v60 = 259;
    v25 = v42;
    v26 = (__int64 *)a1[4];
    *(_QWORD *)v59 = "srem";
    v11 = (_BYTE *)(*(__int64 (__fastcall **)(__int64, __int64, _BYTE *, __int64))(*(_QWORD *)v26[10] + 16LL))(
                     v26[10],
                     23,
                     v42,
                     v24);
    if ( !v11 )
    {
      v63 = 257;
      v11 = (_BYTE *)sub_B504D0(23, (__int64)v25, v24, (__int64)&v61, 0, 0);
      (*(void (__fastcall **)(__int64, _BYTE *, bool *, __int64, __int64))(*(_QWORD *)v26[11] + 16LL))(
        v26[11],
        v11,
        v59,
        v26[7],
        v26[8]);
      v34 = 16LL * *((unsigned int *)v26 + 2);
      v35 = *v26;
      v36 = v35 + v34;
      while ( v35 != v36 )
      {
        v37 = *(_QWORD *)(v35 + 8);
        v38 = *(_DWORD *)v35;
        v35 += 16;
        sub_B99FD0((__int64)v11, v38, v37);
      }
    }
  }
  else
  {
    v60 = 259;
    v27 = v42;
    v28 = (__int64 *)a1[4];
    *(_QWORD *)v59 = "urem";
    v11 = (_BYTE *)(*(__int64 (__fastcall **)(__int64, __int64, _BYTE *, __int64))(*(_QWORD *)v28[10] + 16LL))(
                     v28[10],
                     22,
                     v42,
                     v24);
    if ( !v11 )
    {
      v63 = 257;
      v11 = (_BYTE *)sub_B504D0(22, (__int64)v27, v24, (__int64)&v61, 0, 0);
      (*(void (__fastcall **)(__int64, _BYTE *, bool *, __int64, __int64))(*(_QWORD *)v28[11] + 16LL))(
        v28[11],
        v11,
        v59,
        v28[7],
        v28[8]);
      v29 = 16LL * *((unsigned int *)v28 + 2);
      v30 = *v28;
      v31 = v30 + v29;
      while ( v31 != v30 )
      {
        v32 = *(_QWORD *)(v30 + 8);
        v33 = *(_DWORD *)v30;
        v30 += 16;
        sub_B99FD0((__int64)v11, v33, v32);
      }
    }
  }
  sub_969240((__int64 *)&v57);
  sub_969240((__int64 *)&v55);
LABEL_27:
  if ( v50 > 0x40 && v49 )
    j_j___libc_free_0_0(v49);
  if ( v48 > 0x40 && v47 )
    j_j___libc_free_0_0(v47);
  return v11;
}
