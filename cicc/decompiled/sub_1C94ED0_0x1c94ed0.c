// Function: sub_1C94ED0
// Address: 0x1c94ed0
//
__int64 __fastcall sub_1C94ED0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v12; // rdi
  __int64 v13; // rax
  double v14; // xmm4_8
  double v15; // xmm5_8
  __int64 v16; // r12
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // r10
  __int64 v20; // rbx
  __int64 v21; // r15
  unsigned int v22; // eax
  unsigned int v23; // r12d
  __int64 i; // r14
  _QWORD *v25; // rax
  __int64 v26; // rax
  _BOOL4 v27; // r14d
  char **v28; // rax
  double v29; // xmm4_8
  double v30; // xmm5_8
  __int64 ***v31; // r11
  __int64 v32; // rsi
  _QWORD *v33; // rbx
  _QWORD *v34; // r13
  __int64 v35; // rax
  __int64 v36; // r12
  _QWORD *v37; // rbx
  __int64 v38; // rax
  __int64 v40; // [rsp+0h] [rbp-4E0h]
  char v41; // [rsp+Bh] [rbp-4D5h]
  int v42; // [rsp+Ch] [rbp-4D4h]
  unsigned __int64 v45[2]; // [rsp+30h] [rbp-4B0h] BYREF
  __int64 v46; // [rsp+40h] [rbp-4A0h]
  __int64 v47; // [rsp+50h] [rbp-490h] BYREF
  __int64 v48; // [rsp+58h] [rbp-488h]
  __int64 v49; // [rsp+60h] [rbp-480h]
  int v50; // [rsp+68h] [rbp-478h]
  __int64 v51; // [rsp+70h] [rbp-470h] BYREF
  _BYTE *v52; // [rsp+78h] [rbp-468h]
  _BYTE *v53; // [rsp+80h] [rbp-460h]
  __int64 v54; // [rsp+88h] [rbp-458h]
  int v55; // [rsp+90h] [rbp-450h]
  _BYTE v56[264]; // [rsp+98h] [rbp-448h] BYREF
  _BYTE *v57; // [rsp+1A0h] [rbp-340h] BYREF
  __int64 v58; // [rsp+1A8h] [rbp-338h]
  _BYTE v59[816]; // [rsp+1B0h] [rbp-330h] BYREF

  v52 = v56;
  v53 = v56;
  v57 = v59;
  v58 = 0x2000000000LL;
  v12 = *(_QWORD *)(*a1 + 80);
  v51 = 0;
  v54 = 32;
  v55 = 0;
  if ( v12 )
    v12 -= 24;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v13 = sub_157EE30(v12);
  v16 = *a1;
  v17 = v13;
  v18 = v13 - 24;
  if ( !v17 )
    v18 = 0;
  v40 = v18;
  if ( (*(_BYTE *)(v16 + 18) & 1) != 0 )
  {
    sub_15E08E0(v16, v17);
    v19 = *(_QWORD *)(v16 + 88);
    v20 = v19 + 40LL * *(_QWORD *)(v16 + 96);
    if ( (*(_BYTE *)(v16 + 18) & 1) != 0 )
    {
      sub_15E08E0(v16, v17);
      v19 = *(_QWORD *)(v16 + 88);
    }
  }
  else
  {
    v19 = *(_QWORD *)(v16 + 88);
    v20 = v19 + 40LL * *(_QWORD *)(v16 + 96);
  }
  v21 = v19;
  if ( v20 == v19 )
  {
LABEL_23:
    v33 = v57;
    v34 = &v57[24 * (unsigned int)v58];
    if ( v57 == (_BYTE *)v34 )
    {
      v23 = 1;
      goto LABEL_45;
    }
    while ( 1 )
    {
      v35 = v33[2];
      if ( v35 == 0 || v35 == -8 || v35 == -16 )
        goto LABEL_26;
      v45[0] = 6;
      v45[1] = 0;
      v46 = v33[2];
      v36 = v46;
      if ( v46 != 0 && v46 != -8 && v46 != -16 )
      {
        sub_1649AC0(v45, *v33 & 0xFFFFFFFFFFFFFFF8LL);
        v36 = v46;
      }
      if ( !v36 )
        goto LABEL_26;
      if ( v36 != -16 && v36 != -8 )
        sub_1649B30(v45);
      if ( *(_BYTE *)(v36 + 16) == 77 )
      {
        v33 += 3;
        sub_1AEB420((__int64 ***)v36, 0, a4, a5, a6, a7, v14, v15, a10, a11);
        if ( v34 == v33 )
        {
LABEL_38:
          v23 = 1;
          goto LABEL_39;
        }
      }
      else
      {
        sub_1AEB370(v36, 0);
LABEL_26:
        v33 += 3;
        if ( v34 == v33 )
          goto LABEL_38;
      }
    }
  }
  while ( 1 )
  {
    LOBYTE(v22) = sub_1648CD0(v21, 0);
    v23 = v22;
    if ( (_BYTE)v22 || *(_BYTE *)(*(_QWORD *)v21 + 8LL) != 15 || !(unsigned __int8)sub_15E0450(v21) )
      goto LABEL_9;
    for ( i = *(_QWORD *)(v21 + 8); i; i = *(_QWORD *)(i + 8) )
    {
      v25 = sub_1648700(i);
      if ( *((_BYTE *)v25 + 16) == 72 )
      {
        v26 = *v25;
        if ( *(_BYTE *)(v26 + 8) == 16 )
          v26 = **(_QWORD **)(v26 + 16);
        if ( *(_DWORD *)(v26 + 8) >> 8 == 101 )
          goto LABEL_9;
      }
    }
    v27 = sub_1A018F0(a2, v21);
    v41 = sub_1C2FEA0(*a1, *(_DWORD *)(v21 + 32) + 1);
    v42 = *(_DWORD *)(a1[2] + 8);
    v28 = (char **)sub_1C8DFB0((__int64)a1, v21, (__int64)&v51, v27);
    v31 = (__int64 ***)v28;
    if ( !byte_4FBDBA0 && v42 > 69 && v41 )
    {
      v40 = sub_1C8E770(v28, v40);
      v31 = (__int64 ***)v40;
    }
    if ( !v31 )
      break;
    if ( *(__int64 ***)v21 == *v31 )
    {
      v45[0] = (unsigned __int64)&v51;
      sub_164C7D0(
        v21,
        (__int64)v31,
        (unsigned __int8 (__fastcall *)(__int64, __int64 *))sub_1C8D9F0,
        (__int64)v45,
        a4,
        a5,
        a6,
        a7,
        v29,
        v30,
        a10,
        a11);
LABEL_9:
      v21 += 40;
      if ( v20 == v21 )
        goto LABEL_23;
    }
    else
    {
      v32 = v21;
      v21 += 40;
      sub_1C92200((__int64)a1, v32, v31, (__int64)&v47, a3, (__int64)&v57, a4, a5, a6, a7, v29, v30, a10, a11);
      if ( v20 == v21 )
        goto LABEL_23;
    }
  }
LABEL_39:
  v37 = v57;
  v34 = &v57[24 * (unsigned int)v58];
  if ( v57 != (_BYTE *)v34 )
  {
    do
    {
      v38 = *(v34 - 1);
      v34 -= 3;
      if ( v38 != 0 && v38 != -8 && v38 != -16 )
        sub_1649B30(v34);
    }
    while ( v37 != v34 );
    v34 = v57;
  }
LABEL_45:
  if ( v34 != (_QWORD *)v59 )
    _libc_free((unsigned __int64)v34);
  j___libc_free_0(v48);
  if ( v53 != v52 )
    _libc_free((unsigned __int64)v53);
  return v23;
}
