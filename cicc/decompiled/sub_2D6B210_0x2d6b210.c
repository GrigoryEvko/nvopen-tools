// Function: sub_2D6B210
// Address: 0x2d6b210
//
__int64 __fastcall sub_2D6B210(__int64 a1, __int64 a2, __int64 *a3, unsigned __int8 a4)
{
  __int64 (*v4)(); // rax
  unsigned int v5; // r13d
  unsigned __int8 v10; // r13
  __int64 v11; // r14
  _BYTE *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r13
  unsigned __int8 *v15; // r14
  __int64 v16; // rdi
  unsigned int v17; // r12d
  __int16 v18; // ax
  unsigned __int64 v19; // rax
  unsigned int v20; // eax
  __int64 v21; // rax
  unsigned __int8 *v22; // r10
  _BYTE *v23; // rax
  unsigned int v24; // esi
  __int64 v25; // rsi
  __int64 v26; // r8
  __int64 v27; // r9
  int v28; // eax
  _QWORD *v29; // rax
  unsigned int v30; // eax
  __int64 v31; // r14
  unsigned __int8 *v32; // r10
  _BYTE *v33; // rax
  __int64 v34; // rsi
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rax
  unsigned __int8 v38; // [rsp+23h] [rbp-13Dh]
  __int64 *v39; // [rsp+28h] [rbp-138h]
  __int64 v40; // [rsp+30h] [rbp-130h]
  __int64 v41; // [rsp+30h] [rbp-130h]
  unsigned __int8 *v42; // [rsp+30h] [rbp-130h]
  __int64 v43; // [rsp+38h] [rbp-128h]
  __int64 v44; // [rsp+38h] [rbp-128h]
  __int64 v45; // [rsp+40h] [rbp-120h] BYREF
  unsigned int v46; // [rsp+48h] [rbp-118h]
  unsigned __int64 v47; // [rsp+50h] [rbp-110h] BYREF
  unsigned int v48; // [rsp+58h] [rbp-108h]
  unsigned __int64 v49; // [rsp+60h] [rbp-100h] BYREF
  unsigned int v50; // [rsp+68h] [rbp-F8h]
  __int64 v51; // [rsp+70h] [rbp-F0h] BYREF
  const void **v52; // [rsp+78h] [rbp-E8h] BYREF
  __int16 v53; // [rsp+90h] [rbp-D0h]
  unsigned int *v54; // [rsp+A0h] [rbp-C0h] BYREF
  const void **v55[23]; // [rsp+A8h] [rbp-B8h] BYREF

  v4 = *(__int64 (**)())(*(_QWORD *)a2 + 336LL);
  if ( v4 == sub_2D565D0 )
    return 0;
  v10 = ((__int64 (__fastcall *)(__int64))v4)(a2);
  if ( !v10 )
    return 0;
  if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) != 3 )
    return 0;
  v11 = *(_QWORD *)(a1 - 96);
  if ( *(_BYTE *)v11 != 82 )
    return 0;
  v12 = *(_BYTE **)(v11 - 32);
  if ( *v12 != 17 )
    return 0;
  v13 = *(_QWORD *)(v11 + 16);
  if ( !v13 || *(_QWORD *)(v13 + 8) )
    return 0;
  v40 = *(_QWORD *)(v11 - 64);
  sub_9865C0((__int64)&v45, (__int64)(v12 + 24));
  if ( !*(_QWORD *)(v40 + 16) )
  {
LABEL_43:
    v5 = 0;
    goto LABEL_42;
  }
  v38 = v10;
  v14 = *(_QWORD *)(v40 + 16);
  v43 = v11;
  v39 = a3;
  while ( 1 )
  {
    v15 = *(unsigned __int8 **)(v14 + 24);
    if ( *v15 <= 0x1Cu )
      goto LABEL_14;
    v16 = *((_QWORD *)v15 + 5);
    if ( *(_QWORD *)(a1 + 40) != v16 && (v16 != *(_QWORD *)(a1 - 32) && v16 != *(_QWORD *)(a1 - 64) || !sub_AA54C0(v16)) )
      goto LABEL_14;
    v17 = v46;
    if ( v46 > 0x40 )
    {
      if ( (unsigned int)sub_C44630((__int64)&v45) != 1 )
        goto LABEL_32;
    }
    else if ( !v45 || (v45 & (v45 - 1)) != 0 )
    {
      goto LABEL_32;
    }
    v18 = *(_WORD *)(v43 + 2);
    if ( (v18 & 0x3F) != 0x24 )
      goto LABEL_21;
    v28 = sub_9871A0((__int64)&v45);
    v54 = (unsigned int *)v40;
    v55[0] = (const void **)(v17 - 1 - v28);
    if ( (unsigned int)*v15 - 55 <= 1 )
    {
      v29 = (_QWORD *)sub_986520((__int64)v15);
      if ( v40 == *v29 )
      {
        LOBYTE(v30) = sub_F17ED0(v55, v29[4]);
        if ( (_BYTE)v30 )
          break;
      }
    }
LABEL_32:
    v18 = *(_WORD *)(v43 + 2);
LABEL_21:
    if ( (v18 & 0x3Fu) - 32 <= 1 )
    {
      sub_9865C0((__int64)&v47, (__int64)&v45);
      if ( v48 > 0x40 )
      {
        sub_C43D10((__int64)&v47);
      }
      else
      {
        v19 = 0;
        if ( v48 )
          v19 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v48;
        v47 = ~v47 & v19;
      }
      sub_C46250((__int64)&v47);
      v20 = v48;
      v48 = 0;
      v52 = (const void **)&v49;
      v50 = v20;
      v49 = v47;
      v51 = v40;
      if ( *v15 == 42 && (v21 = *((_QWORD *)v15 - 8)) != 0 && v40 == v21 && sub_10080A0(&v52, *((_QWORD *)v15 - 4))
        || (v55[0] = (const void **)&v45, v54 = (unsigned int *)v40, *v15 == 44)
        && (v37 = *((_QWORD *)v15 - 8)) != 0
        && v40 == v37
        && sub_10080A0(v55, *((_QWORD *)v15 - 4)) )
      {
        v5 = v38;
        sub_969240((__int64 *)&v49);
        sub_969240((__int64 *)&v47);
        sub_23D0AB0((__int64)&v54, a1, 0, 0, 0);
        v22 = v15;
        if ( *(_QWORD *)(a1 + 40) != *((_QWORD *)v15 + 5) )
        {
          sub_B444E0(v15, a1 + 24, 0);
          v22 = v15;
        }
        v41 = (__int64)v22;
        sub_B44F30(v22);
        v53 = 257;
        v23 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v41 + 8), 0, 0);
        v24 = *(_WORD *)(v43 + 2) & 0x3F;
        if ( (*(_WORD *)(v43 + 2) & 0x30) != 0 )
        {
          v25 = sub_92B530(&v54, v24, v41, v23, (__int64)&v51);
        }
        else
        {
          HIDWORD(v49) = 0;
          v25 = sub_B35C90((__int64)&v54, v24, v41, (__int64)v23, (__int64)&v51, 0, (unsigned int)v49, 0);
        }
        sub_2D594F0(v43, v25, v39, a4, v26, v27);
        sub_F94A20(&v54, v25);
        goto LABEL_42;
      }
      sub_969240((__int64 *)&v49);
      sub_969240((__int64 *)&v47);
    }
LABEL_14:
    v14 = *(_QWORD *)(v14 + 8);
    if ( !v14 )
      goto LABEL_43;
  }
  v5 = v30;
  v42 = v15;
  v31 = v43;
  sub_23D0AB0((__int64)&v54, a1, 0, 0, 0);
  v32 = v42;
  if ( *(_QWORD *)(a1 + 40) != *((_QWORD *)v42 + 5) )
  {
    sub_B444E0(v42, a1 + 24, 0);
    v32 = v42;
  }
  v44 = (__int64)v32;
  sub_B44F30(v32);
  v53 = 257;
  v33 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v44 + 8), 0, 0);
  v34 = sub_92B530(&v54, 0x20u, v44, v33, (__int64)&v51);
  sub_2D594F0(v31, v34, v39, a4, v35, v36);
  sub_F94A20(&v54, v34);
LABEL_42:
  sub_969240(&v45);
  return v5;
}
