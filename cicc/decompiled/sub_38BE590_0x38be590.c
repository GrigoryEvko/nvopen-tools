// Function: sub_38BE590
// Address: 0x38be590
//
__int64 __fastcall sub_38BE590(
        __int64 a1,
        unsigned __int8 *a2,
        size_t a3,
        int a4,
        int a5,
        unsigned __int8 a6,
        int a7,
        __int64 a8,
        int a9,
        __int64 a10)
{
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  _QWORD *v15; // r9
  __int64 v16; // r13
  _QWORD *v17; // r15
  unsigned __int64 v18; // rax
  unsigned int v19; // r9d
  __int64 *v20; // r11
  __int64 v21; // rsi
  _QWORD *v22; // rax
  unsigned __int64 v23; // rax
  __int64 v24; // r13
  __int64 v25; // rax
  __int64 v26; // rbx
  __int64 v27; // r12
  __int64 *v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v32; // rax
  unsigned int v33; // r8d
  _QWORD *v34; // r9
  _QWORD *v35; // r13
  unsigned int v36; // eax
  __int64 *v37; // rax
  __int64 *v38; // rax
  __int64 v39; // rax
  unsigned int v40; // r9d
  __int64 *v41; // r11
  __int64 v42; // r8
  __int64 *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // [rsp+8h] [rbp-78h]
  _QWORD *v48; // [rsp+10h] [rbp-70h]
  __int64 *v49; // [rsp+10h] [rbp-70h]
  unsigned int v50; // [rsp+18h] [rbp-68h]
  unsigned int v51; // [rsp+18h] [rbp-68h]
  const char *v55; // [rsp+30h] [rbp-50h] BYREF
  char v56; // [rsp+40h] [rbp-40h]
  char v57; // [rsp+41h] [rbp-3Fh]

  v12 = (unsigned int)sub_16D19C0(a1 + 568, a2, a3);
  v14 = v12;
  v15 = (_QWORD *)(*(_QWORD *)(a1 + 568) + 8 * v12);
  v16 = *v15;
  if ( *v15 )
  {
    if ( v16 != -8 )
      goto LABEL_3;
    --*(_DWORD *)(a1 + 584);
  }
  v48 = v15;
  v50 = v12;
  v32 = sub_145CBF0(*(__int64 **)(a1 + 592), a3 + 17, 8);
  v33 = v50;
  v34 = v48;
  v35 = (_QWORD *)v32;
  if ( a3 + 1 > 1 )
  {
    memcpy((void *)(v32 + 16), a2, a3);
    v34 = v48;
    v33 = v50;
  }
  *((_BYTE *)v35 + a3 + 16) = 0;
  *v35 = a3;
  v35[1] = 0;
  *v34 = v35;
  ++*(_DWORD *)(a1 + 580);
  v36 = sub_16D1CD0(a1 + 568, v33);
  v12 = *(_QWORD *)(a1 + 568);
  v37 = (__int64 *)(v12 + 8LL * v36);
  v16 = *v37;
  if ( !*v37 || v16 == -8 )
  {
    v38 = v37 + 1;
    do
    {
      do
        v16 = *v38++;
      while ( v16 == -8 );
    }
    while ( !v16 );
  }
LABEL_3:
  v17 = *(_QWORD **)(v16 + 8);
  if ( !v17 )
    goto LABEL_8;
  v18 = *v17 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v18 )
    goto LABEL_43;
  if ( (*((_BYTE *)v17 + 9) & 0xC) == 8 )
  {
    *((_BYTE *)v17 + 8) |= 4u;
    v12 = sub_38CE440(v17[3]);
    *v17 = v12 | *v17 & 7LL;
    v17 = *(_QWORD **)(v16 + 8);
    if ( !v12 )
      goto LABEL_50;
    v18 = *v17 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v18 )
    {
      if ( (*((_BYTE *)v17 + 9) & 0xC) != 8 )
        goto LABEL_49;
      *((_BYTE *)v17 + 8) |= 4u;
      v44 = sub_38CE440(v17[3]);
      v45 = v44 | *v17 & 7LL;
      *v17 = v45;
      if ( !v44 )
        goto LABEL_49;
      v18 = v45 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v18 )
      {
        v18 = 0;
        if ( (*((_BYTE *)v17 + 9) & 0xC) == 8 )
        {
          *((_BYTE *)v17 + 8) |= 4u;
          v18 = sub_38CE440(v17[3]);
          *v17 = v18 | *v17 & 7LL;
        }
      }
    }
LABEL_43:
    v12 = (__int64)&off_4CF6DB8;
    if ( off_4CF6DB8 != (_UNKNOWN *)v18 )
    {
      v17 = *(_QWORD **)(v16 + 8);
      if ( (*v17 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        if ( v17 == *(_QWORD **)(*(_QWORD *)((*v17 & 0xFFFFFFFFFFFFFFF8LL) + 24) + 8LL) )
          goto LABEL_51;
      }
      else
      {
        if ( (*((_BYTE *)v17 + 9) & 0xC) != 8 )
          BUG();
        *((_BYTE *)v17 + 8) |= 4u;
        v46 = sub_38CE440(v17[3]);
        v12 = v46 | *v17 & 7LL;
        *v17 = v12;
        if ( !v46 )
          BUG();
        v17 = *(_QWORD **)(v16 + 8);
        if ( *(_QWORD **)(*(_QWORD *)(v46 + 24) + 8LL) == v17 )
        {
LABEL_50:
          if ( !v17 )
            goto LABEL_8;
LABEL_51:
          if ( (*v17 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
            goto LABEL_8;
          goto LABEL_6;
        }
      }
    }
LABEL_49:
    v57 = 1;
    v55 = "invalid symbol redefinition";
    v56 = 3;
    sub_38BE3D0(a1, 0, (__int64)&v55);
    v17 = *(_QWORD **)(v16 + 8);
    goto LABEL_50;
  }
LABEL_6:
  if ( (*((_BYTE *)v17 + 9) & 0xC) != 8 )
    goto LABEL_14;
  *((_BYTE *)v17 + 8) |= 4u;
  v12 = sub_38CE440(v17[3]);
  *v17 = v12 | *v17 & 7LL;
  if ( !v12 )
  {
    v17 = *(_QWORD **)(v16 + 8);
    goto LABEL_14;
  }
LABEL_8:
  v19 = sub_16D19C0(a1 + 632, a2, a3);
  v20 = (__int64 *)(*(_QWORD *)(a1 + 632) + 8LL * v19);
  v21 = *v20;
  if ( !*v20 )
  {
LABEL_29:
    v49 = v20;
    v51 = v19;
    v39 = sub_145CBF0(*(__int64 **)(a1 + 656), a3 + 17, 8);
    v40 = v51;
    v41 = v49;
    v42 = v39;
    if ( a3 + 1 > 1 )
    {
      v47 = v39;
      memcpy((void *)(v39 + 16), a2, a3);
      v42 = v47;
      v41 = v49;
      v40 = v51;
    }
    *(_BYTE *)(v42 + a3 + 16) = 0;
    *(_QWORD *)v42 = a3;
    *(_BYTE *)(v42 + 8) = 0;
    *v41 = v42;
    ++*(_DWORD *)(a1 + 644);
    v43 = (__int64 *)(*(_QWORD *)(a1 + 632) + 8LL * (unsigned int)sub_16D1CD0(a1 + 632, v40));
    v21 = *v43;
    if ( !*v43 || v21 == -8 )
    {
      do
      {
        do
        {
          v21 = v43[1];
          ++v43;
        }
        while ( v21 == -8 );
      }
      while ( !v21 );
    }
    goto LABEL_10;
  }
  if ( v21 == -8 )
  {
    --*(_DWORD *)(a1 + 648);
    goto LABEL_29;
  }
LABEL_10:
  v22 = (_QWORD *)sub_38E22B0(40, v21, a1);
  v17 = v22;
  if ( v22 )
  {
    v22[3] = 0;
    *v22 = 4;
    v23 = v22[1] & 0xFFFF0000FFFE0000LL;
    LOBYTE(v23) = 0x80;
    *((_DWORD *)v17 + 4) = 0;
    v17[1] = v23;
    *(v17 - 1) = v21;
    v17[4] = 0;
  }
  if ( !*(_QWORD *)(v16 + 8) )
    *(_QWORD *)(v16 + 8) = v17;
LABEL_14:
  sub_38E2920(v17, 0, v12, v13, v14);
  sub_38E28A0(v17, 3);
  v24 = sub_145CBF0((__int64 *)(a1 + 256), 200, 8);
  sub_38D76F0(v24, 1, a6, v17);
  *(_QWORD *)(v24 + 152) = a2;
  *(_QWORD *)(v24 + 160) = a3;
  *(_QWORD *)v24 = &unk_4A3E598;
  *(_DWORD *)(v24 + 168) = a4;
  *(_DWORD *)(v24 + 172) = a5;
  *(_DWORD *)(v24 + 176) = a9;
  *(_DWORD *)(v24 + 180) = a7;
  *(_QWORD *)(v24 + 184) = a8;
  *(_QWORD *)(v24 + 192) = a10;
  if ( a8 )
    sub_38E2780(a8);
  v25 = sub_22077B0(0xE0u);
  v26 = v25;
  if ( v25 )
  {
    v27 = v25;
    sub_38CF760(v25, 1, 0, 0);
    *(_QWORD *)(v26 + 56) = 0;
    *(_WORD *)(v26 + 48) = 0;
    *(_QWORD *)(v26 + 64) = v26 + 80;
    *(_QWORD *)(v26 + 72) = 0x2000000000LL;
    *(_QWORD *)(v26 + 112) = v26 + 128;
    *(_QWORD *)(v26 + 120) = 0x400000000LL;
  }
  else
  {
    v27 = 0;
  }
  v28 = *(__int64 **)(v24 + 104);
  v29 = *v28;
  v30 = *(_QWORD *)v26 & 7LL;
  *(_QWORD *)(v26 + 8) = v28;
  v29 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v26 = v29 | v30;
  *(_QWORD *)(v29 + 8) = v27;
  *v28 = *v28 & 7 | v27;
  *(_QWORD *)(v26 + 24) = v24;
  *v17 = *v17 & 7LL | v26;
  return v24;
}
