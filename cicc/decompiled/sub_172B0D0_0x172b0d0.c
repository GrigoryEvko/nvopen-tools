// Function: sub_172B0D0
// Address: 0x172b0d0
//
unsigned __int8 *__fastcall sub_172B0D0(__int64 a1, __int64 a2, char a3, __int64 a4, double a5, double a6, double a7)
{
  __int64 *v7; // r15
  _BYTE *v9; // rdi
  unsigned __int8 v10; // al
  __int64 v11; // r14
  _BYTE *v12; // rdi
  unsigned __int8 v13; // al
  __int64 v14; // r13
  int v15; // eax
  int v16; // ebx
  __int64 v17; // rax
  unsigned int v18; // edx
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rax
  unsigned int v21; // ebx
  unsigned int v22; // edx
  const void *v23; // r9
  bool v24; // al
  bool v25; // bl
  unsigned __int8 *v26; // r9
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  unsigned __int8 *v31; // rax
  __int64 v32; // r12
  __int64 v33; // rax
  _QWORD *v34; // rdx
  _QWORD *v35; // rsi
  unsigned __int16 v36; // di
  __int64 v37; // rax
  unsigned int v38; // ebx
  __int64 v39; // rax
  unsigned int v40; // ecx
  unsigned __int64 v41; // rdx
  unsigned int v42; // eax
  __int64 v43; // rax
  unsigned __int8 *v44; // rax
  unsigned __int8 *v45; // r9
  __int64 v46; // rax
  const void *v47; // [rsp+0h] [rbp-A0h]
  unsigned __int8 *v48; // [rsp+0h] [rbp-A0h]
  unsigned __int8 *v49; // [rsp+0h] [rbp-A0h]
  __int64 v50; // [rsp+0h] [rbp-A0h]
  unsigned int v51; // [rsp+8h] [rbp-98h]
  unsigned __int8 *v54; // [rsp+18h] [rbp-88h]
  __int64 v55; // [rsp+18h] [rbp-88h]
  unsigned __int64 v56; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v57; // [rsp+28h] [rbp-78h]
  unsigned __int64 v58; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v59; // [rsp+38h] [rbp-68h]
  const void *v60; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v61; // [rsp+48h] [rbp-58h]
  unsigned __int64 v62; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v63; // [rsp+58h] [rbp-48h]
  __int16 v64; // [rsp+60h] [rbp-40h]

  v7 = *(__int64 **)(a1 - 48);
  if ( v7 != *(__int64 **)(a2 - 48) )
    return 0;
  v9 = *(_BYTE **)(a1 - 24);
  v10 = v9[16];
  v11 = (__int64)(v9 + 24);
  if ( v10 != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) != 16 )
      return 0;
    if ( v10 > 0x10u )
      return 0;
    v29 = sub_15A1020(v9, a2, *(_QWORD *)v9, a4);
    if ( !v29 || *(_BYTE *)(v29 + 16) != 13 )
      return 0;
    v11 = v29 + 24;
  }
  v12 = *(_BYTE **)(a2 - 24);
  v13 = v12[16];
  v14 = (__int64)(v12 + 24);
  if ( v13 == 13 )
    goto LABEL_4;
  if ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) != 16 )
    return 0;
  if ( v13 > 0x10u )
    return 0;
  v28 = sub_15A1020(v12, a2, *(_QWORD *)v12, a4);
  if ( !v28 || *(_BYTE *)(v28 + 16) != 13 )
    return 0;
  v14 = v28 + 24;
LABEL_4:
  v15 = *(unsigned __int16 *)(a2 + 18);
  v16 = *(unsigned __int16 *)(a1 + 18);
  BYTE1(v15) &= ~0x80u;
  BYTE1(v16) &= ~0x80u;
  if ( v15 != v16 || v16 != 33 && a3 || a3 != 1 && v16 != 32 )
    return 0;
  if ( (int)sub_16A9900(v11, (unsigned __int64 *)v14) > 0 )
  {
    v17 = v11;
    v11 = v14;
    v14 = v17;
  }
  v18 = *(_DWORD *)(v11 + 8);
  v63 = v18;
  if ( v18 <= 0x40 )
  {
    v19 = *(_QWORD *)v11;
    v62 = *(_QWORD *)v11;
LABEL_13:
    v20 = *(_QWORD *)v14 ^ v19;
    v57 = v18;
    v56 = v20;
    goto LABEL_14;
  }
  sub_16A4FD0((__int64)&v62, (const void **)v11);
  v18 = v63;
  if ( v63 <= 0x40 )
  {
    v19 = v62;
    goto LABEL_13;
  }
  sub_16A8F00((__int64 *)&v62, (__int64 *)v14);
  v20 = v62;
  v57 = v63;
  v56 = v62;
  if ( v63 > 0x40 )
  {
    if ( (unsigned int)sub_16A5940((__int64)&v56) != 1 )
      goto LABEL_16;
    goto LABEL_46;
  }
LABEL_14:
  if ( !v20 || (v20 & (v20 - 1)) != 0 )
  {
LABEL_16:
    v21 = *(_DWORD *)(v11 + 8);
    if ( v21 <= 0x40 )
    {
      if ( *(_QWORD *)v11 )
        goto LABEL_18;
    }
    else if ( v21 != (unsigned int)sub_16A57B0(v11) )
    {
      goto LABEL_18;
    }
    v38 = *(_DWORD *)(v14 + 8);
    if ( v38 <= 0x40 )
    {
      if ( *(_QWORD *)v14 != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v38) )
      {
LABEL_18:
        v61 = *(_DWORD *)(v14 + 8);
        if ( v61 > 0x40 )
          sub_16A4FD0((__int64)&v60, (const void **)v14);
        else
          v60 = *(const void **)v14;
        sub_16A7800((__int64)&v60, 1u);
        v22 = v61;
        v23 = v60;
        v61 = 0;
        v63 = v22;
        v62 = (unsigned __int64)v60;
        if ( *(_DWORD *)(v11 + 8) <= 0x40u )
        {
          v25 = *(_QWORD *)v11 == (_QWORD)v60;
        }
        else
        {
          v47 = v60;
          v51 = v22;
          v24 = sub_16A5220(v11, (const void **)&v62);
          v23 = v47;
          v22 = v51;
          v25 = v24;
        }
        if ( v22 > 0x40 )
        {
          if ( v23 )
          {
            j_j___libc_free_0_0(v23);
            if ( v61 > 0x40 )
            {
              if ( v60 )
                j_j___libc_free_0_0(v60);
            }
          }
        }
        v26 = 0;
        if ( !v25 )
          goto LABEL_28;
        v64 = 257;
        v40 = *(_DWORD *)(v11 + 8);
        v59 = v40;
        if ( v40 > 0x40 )
        {
          sub_16A4FD0((__int64)&v58, (const void **)v11);
          LOBYTE(v40) = v59;
          if ( v59 > 0x40 )
          {
            sub_16A8F40((__int64 *)&v58);
LABEL_64:
            sub_16A7400((__int64)&v58);
            v42 = v59;
            v59 = 0;
            v61 = v42;
            v60 = (const void *)v58;
            v43 = sub_15A1070(*v7, (__int64)&v60);
            v44 = sub_17094A0(a4, (__int64)v7, v43, (__int64 *)&v62, 0, 0, a5, a6, a7);
            v45 = v44;
            if ( v61 > 0x40 && v60 )
            {
              v48 = v44;
              j_j___libc_free_0_0(v60);
              v45 = v48;
            }
            if ( v59 > 0x40 && v58 )
            {
              v49 = v45;
              j_j___libc_free_0_0(v58);
              v45 = v49;
            }
            v50 = (__int64)v45;
            v64 = 257;
            v46 = sub_15A0680(*v7, 1, 0);
            v34 = (_QWORD *)v46;
            if ( *(_BYTE *)(v50 + 16) <= 0x10u && *(_BYTE *)(v46 + 16) <= 0x10u )
            {
              v35 = (_QWORD *)v50;
              v36 = a3 == 0 ? 37 : 34;
LABEL_49:
              v55 = sub_15A37B0(v36, v35, v34, 0);
              v37 = sub_14DBA30(v55, *(_QWORD *)(a4 + 96), 0);
              v26 = (unsigned __int8 *)v55;
              if ( v37 )
                v26 = (unsigned __int8 *)v37;
              goto LABEL_28;
            }
            v26 = sub_1727440(a4, a3 == 0 ? 37 : 34, v50, v46, (__int64 *)&v62);
            goto LABEL_28;
          }
          v41 = v58;
        }
        else
        {
          v41 = *(_QWORD *)v11;
        }
        v58 = ~v41 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v40);
        goto LABEL_64;
      }
    }
    else if ( v38 != (unsigned int)sub_16A58F0(v14) )
    {
      goto LABEL_18;
    }
    v39 = v11;
    v11 = v14;
    v14 = v39;
    goto LABEL_18;
  }
LABEL_46:
  v64 = 257;
  v30 = sub_15A1070(*v7, (__int64)&v56);
  v31 = sub_172AC10(a4, (__int64)v7, v30, (__int64 *)&v62, a5, a6, a7);
  v64 = 257;
  v32 = (__int64)v31;
  v33 = sub_15A1070(*v7, v14);
  v34 = (_QWORD *)v33;
  if ( *(_BYTE *)(v32 + 16) <= 0x10u && *(_BYTE *)(v33 + 16) <= 0x10u )
  {
    v35 = (_QWORD *)v32;
    v36 = v16;
    goto LABEL_49;
  }
  v26 = sub_1727440(a4, v16, v32, v33, (__int64 *)&v62);
LABEL_28:
  if ( v57 > 0x40 && v56 )
  {
    v54 = v26;
    j_j___libc_free_0_0(v56);
    return v54;
  }
  return v26;
}
