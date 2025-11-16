// Function: sub_111C5E0
// Address: 0x111c5e0
//
_QWORD *__fastcall sub_111C5E0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  _QWORD *v4; // r14
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int16 v8; // cx
  __int64 v10; // rax
  unsigned int v11; // eax
  unsigned int v12; // r8d
  __int64 v13; // rcx
  __int64 v14; // rdi
  int v15; // eax
  bool v16; // al
  unsigned int v17; // eax
  unsigned int v18; // edx
  unsigned int v19; // eax
  unsigned __int64 v20; // rdi
  __int64 v21; // rsi
  __int64 v22; // rax
  unsigned __int8 *v23; // rcx
  bool v24; // dl
  int v25; // eax
  _BYTE *v26; // rax
  _BYTE *v27; // r12
  unsigned int **v28; // r14
  _BYTE *v29; // rax
  __int64 v30; // rbx
  __int64 v31; // rax
  __int16 v32; // r15
  __int64 v33; // r12
  _QWORD **v34; // rdx
  int v35; // ecx
  __int64 *v36; // rax
  __int64 v37; // rsi
  _BYTE *v38; // rax
  int v39; // [rsp+0h] [rbp-E0h]
  unsigned int v40; // [rsp+4h] [rbp-DCh]
  int v41; // [rsp+4h] [rbp-DCh]
  unsigned int v42; // [rsp+4h] [rbp-DCh]
  bool v43; // [rsp+8h] [rbp-D8h]
  unsigned __int8 *v44; // [rsp+8h] [rbp-D8h]
  int v45; // [rsp+10h] [rbp-D0h]
  unsigned int v46; // [rsp+10h] [rbp-D0h]
  unsigned __int8 *v47; // [rsp+10h] [rbp-D0h]
  __int64 v48; // [rsp+10h] [rbp-D0h]
  unsigned int v49; // [rsp+18h] [rbp-C8h]
  unsigned int v50; // [rsp+18h] [rbp-C8h]
  unsigned __int8 *v51; // [rsp+18h] [rbp-C8h]
  unsigned int v52; // [rsp+18h] [rbp-C8h]
  char v53; // [rsp+18h] [rbp-C8h]
  __int64 v54; // [rsp+18h] [rbp-C8h]
  const void **v55; // [rsp+20h] [rbp-C0h]
  __int64 v57; // [rsp+38h] [rbp-A8h]
  _BYTE v58[32]; // [rsp+40h] [rbp-A0h] BYREF
  __int16 v59; // [rsp+60h] [rbp-80h]
  unsigned __int64 v60; // [rsp+70h] [rbp-70h] BYREF
  unsigned int v61; // [rsp+78h] [rbp-68h]
  unsigned __int64 v62; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v63; // [rsp+88h] [rbp-58h]
  __int64 v64; // [rsp+90h] [rbp-50h] BYREF
  unsigned int v65; // [rsp+98h] [rbp-48h]
  char v66; // [rsp+A0h] [rbp-40h]

  v2 = *(_QWORD *)(a2 - 64);
  v3 = *(_QWORD *)(v2 + 16);
  if ( !v3 )
    return 0;
  v4 = *(_QWORD **)(v3 + 8);
  if ( v4 )
    return 0;
  if ( *(_BYTE *)v2 != 67 )
    return 0;
  v6 = *(_QWORD *)(v2 - 32);
  if ( !v6 )
    return 0;
  v7 = *(_QWORD *)(a2 - 32);
  v8 = *(_WORD *)(a2 + 2);
  if ( *(_BYTE *)v7 == 17 )
  {
    v55 = (const void **)(v7 + 24);
    goto LABEL_9;
  }
  v53 = *(_WORD *)(a2 + 2);
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v7 + 8) + 8LL) - 17 > 1 )
    return 0;
  if ( *(_BYTE *)v7 > 0x15u )
    return 0;
  v48 = *(_QWORD *)(a2 - 32);
  v26 = sub_AD7630(v7, 0, v7);
  if ( !v26 || *v26 != 17 )
    return 0;
  v7 = v48;
  LOBYTE(v8) = v53;
  v55 = (const void **)(v26 + 24);
LABEL_9:
  sub_11FB020(&v60, v2, v7, v8 & 0x3F, 1, 1);
  if ( v66 )
  {
    v27 = (_BYTE *)v60;
    v28 = (unsigned int **)a1[4];
    v59 = 257;
    v29 = (_BYTE *)sub_AD8D80(*(_QWORD *)(v60 + 8), (__int64)&v62);
    v30 = sub_A82350(v28, v27, v29, (__int64)v58);
    v31 = sub_AD8D80(*(_QWORD *)(v60 + 8), (__int64)&v64);
    v32 = v61;
    v33 = v31;
    v59 = 257;
    v4 = sub_BD2C40(72, unk_3F10FD0);
    if ( v4 )
    {
      v34 = *(_QWORD ***)(v30 + 8);
      v35 = *((unsigned __int8 *)v34 + 8);
      if ( (unsigned int)(v35 - 17) > 1 )
      {
        v37 = sub_BCB2A0(*v34);
      }
      else
      {
        BYTE4(v57) = (_BYTE)v35 == 18;
        LODWORD(v57) = *((_DWORD *)v34 + 8);
        v36 = (__int64 *)sub_BCB2A0(*v34);
        v37 = sub_BCE1B0(v36, v57);
      }
      sub_B523C0((__int64)v4, v37, 53, v32, v30, v33, (__int64)v58, 0, 0, 0);
    }
    if ( v66 )
    {
      v66 = 0;
      if ( v65 > 0x40 && v64 )
        j_j___libc_free_0_0(v64);
      if ( v63 > 0x40 )
      {
        v20 = v62;
        if ( v62 )
          goto LABEL_26;
      }
    }
    return v4;
  }
  if ( *(_BYTE *)v6 != 85 )
    return v4;
  v10 = *(_QWORD *)(v6 - 32);
  if ( !v10
    || *(_BYTE *)v10
    || *(_QWORD *)(v10 + 24) != *(_QWORD *)(v6 + 80)
    || (*(_BYTE *)(v10 + 33) & 0x20) == 0
    || (*(_DWORD *)(v10 + 36) & 0xFFFFFFFD) != 0x41 )
  {
    return v4;
  }
  v11 = sub_BCB060(*(_QWORD *)(v6 + 8));
  v12 = v11;
  v13 = *(_QWORD *)(v6 + 32 * (1LL - (*(_DWORD *)(v6 + 4) & 0x7FFFFFF)));
  if ( *(_BYTE *)v13 == 17 )
  {
    if ( *(_DWORD *)(v13 + 32) > 0x40u )
    {
      v45 = *(_DWORD *)(v13 + 32);
      v14 = v13 + 24;
      v49 = v11;
LABEL_19:
      v15 = sub_C444A0(v14);
      v12 = v49;
      v16 = v45 - 1 == v15;
      goto LABEL_20;
    }
    v16 = *(_QWORD *)(v13 + 24) == 1;
  }
  else
  {
    v54 = *(_QWORD *)(v13 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v54 + 8) - 17 > 1 || *(_BYTE *)v13 > 0x15u )
      goto LABEL_38;
    v42 = v11;
    v44 = *(unsigned __int8 **)(v6 + 32 * (1LL - (*(_DWORD *)(v6 + 4) & 0x7FFFFFF)));
    v38 = sub_AD7630(v13, 0, 0);
    v23 = v44;
    v12 = v42;
    v24 = 0;
    if ( !v38 || *v38 != 17 )
    {
      if ( *(_BYTE *)(v54 + 8) == 17 )
      {
        v39 = *(_DWORD *)(v54 + 32);
        if ( v39 )
        {
          v21 = 0;
          while ( 1 )
          {
            v40 = v12;
            v43 = v24;
            v51 = v23;
            v22 = sub_AD69F0(v23, v21);
            v12 = v40;
            if ( !v22 )
              break;
            v23 = v51;
            v24 = v43;
            if ( *(_BYTE *)v22 != 13 )
            {
              if ( *(_BYTE *)v22 != 17 )
                break;
              if ( *(_DWORD *)(v22 + 32) <= 0x40u )
              {
                v24 = *(_QWORD *)(v22 + 24) == 1;
              }
              else
              {
                v41 = *(_DWORD *)(v22 + 32);
                v47 = v51;
                v52 = v12;
                v25 = sub_C444A0(v22 + 24);
                v23 = v47;
                v12 = v52;
                v24 = v41 - 1 == v25;
              }
              if ( !v24 )
                break;
            }
            v21 = (unsigned int)(v21 + 1);
            if ( v39 == (_DWORD)v21 )
            {
              if ( v24 )
                goto LABEL_21;
              goto LABEL_38;
            }
          }
        }
      }
      goto LABEL_38;
    }
    if ( *((_DWORD *)v38 + 8) > 0x40u )
    {
      v45 = *((_DWORD *)v38 + 8);
      v14 = (__int64)(v38 + 24);
      v49 = v42;
      goto LABEL_19;
    }
    v16 = *((_QWORD *)v38 + 3) == 1;
  }
LABEL_20:
  if ( v16 )
  {
LABEL_21:
    v17 = v12 - 1;
    goto LABEL_22;
  }
LABEL_38:
  v17 = v12;
LABEL_22:
  if ( !v17
    || (_BitScanReverse(&v18, v17),
        v46 = v12,
        v50 = v18 ^ 0x1F,
        v19 = sub_BCB060(*(_QWORD *)(v2 + 8)),
        v12 = v46,
        32 - v50 <= v19) )
  {
    sub_C449B0((__int64)&v60, v55, v12);
    v4 = sub_111AB00(a1, a2, v6, &v60);
    if ( v61 > 0x40 )
    {
      v20 = v60;
      if ( v60 )
LABEL_26:
        j_j___libc_free_0_0(v20);
    }
  }
  return v4;
}
