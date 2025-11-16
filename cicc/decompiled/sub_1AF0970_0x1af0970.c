// Function: sub_1AF0970
// Address: 0x1af0970
//
void __fastcall sub_1AF0970(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rdx
  __int64 v12; // rcx
  int v13; // r8d
  int v14; // r9d
  unsigned __int64 v15; // r14
  double v16; // xmm4_8
  double v17; // xmm5_8
  char v18; // al
  __int64 v19; // r15
  _QWORD *v20; // rax
  __int64 v21; // r12
  __int64 v22; // rbx
  __int64 v23; // rsi
  __int64 *v24; // r8
  double v25; // xmm4_8
  double v26; // xmm5_8
  __int64 v27; // rsi
  unsigned __int8 *v28; // rsi
  __int64 v29; // rdx
  int v30; // ebx
  int v31; // ebx
  __int64 **v32; // rax
  __int64 v33; // rax
  char v34; // al
  unsigned __int64 v35; // rdx
  __int64 *v36; // rdx
  __int64 *v37; // rbx
  __int64 v38; // rdi
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  unsigned __int64 v44; // rax
  __int64 *v46; // [rsp+8h] [rbp-68h]
  __int64 *v47; // [rsp+8h] [rbp-68h]
  __int64 *v48; // [rsp+8h] [rbp-68h]
  _QWORD v49[2]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v50[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v51; // [rsp+30h] [rbp-40h]

  v15 = sub_157EBA0(a1);
  v18 = *(_BYTE *)(v15 + 16);
  if ( v18 == 29 )
  {
    sub_1AF0320(v15, a2, a3, a4, a5, a6, v16, v17, a9, a10, v11, v12, v13, v14);
    return;
  }
  if ( v18 != 32 )
  {
    v49[0] = sub_1649960(v15);
    v51 = 261;
    v49[1] = v29;
    v50[0] = (__int64)v49;
    v30 = *(_DWORD *)(v15 + 20);
    if ( (*(_BYTE *)(v15 + 18) & 1) != 0 )
      v31 = (v30 & 0xFFFFFFF) - 2;
    else
      v31 = (v30 & 0xFFFFFFF) - 1;
    if ( (*(_BYTE *)(v15 + 23) & 0x40) != 0 )
      v32 = *(__int64 ***)(v15 - 8);
    else
      v32 = (__int64 **)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
    v47 = *v32;
    v33 = sub_1648B60(64);
    v21 = v33;
    if ( v33 )
      sub_15F7B50(v33, v47, 0, v31, (__int64)v50, v15);
    v34 = *(_BYTE *)(v15 + 23);
    if ( (v34 & 0x40) != 0 )
    {
      v35 = *(_QWORD *)(v15 - 8);
      v48 = (__int64 *)(v35 + 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
    }
    else
    {
      v48 = (__int64 *)v15;
      v35 = v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF);
    }
    if ( (*(_BYTE *)(v15 + 18) & 1) != 0 )
    {
      v36 = (__int64 *)(v35 + 48);
      if ( v36 == v48 )
      {
LABEL_35:
        if ( (v34 & 0x40) != 0 )
          v44 = *(_QWORD *)(v15 - 8);
        else
          v44 = v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF);
        v22 = *(_QWORD *)(v44 + 24);
        goto LABEL_8;
      }
    }
    else
    {
      v36 = (__int64 *)(v35 + 24);
      if ( v36 == v48 )
      {
        v22 = 0;
        goto LABEL_8;
      }
    }
    v37 = v36;
    do
    {
      v38 = *v37;
      v37 += 3;
      v39 = sub_15A5110(v38);
      sub_15F7DB0(v21, v39, v40, v41, v42, v43);
    }
    while ( v48 != v37 );
    v22 = 0;
    if ( (*(_BYTE *)(v15 + 18) & 1) == 0 )
      goto LABEL_8;
    v34 = *(_BYTE *)(v15 + 23);
    goto LABEL_35;
  }
  v19 = *(_QWORD *)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
  v20 = sub_1648A60(56, 1u);
  v21 = (__int64)v20;
  if ( v20 )
    sub_15F76D0((__int64)v20, v19, 0, 1u, v15);
  if ( (*(_BYTE *)(v15 + 18) & 1) != 0 )
    v22 = *(_QWORD *)(v15 + 24 * (1LL - (*(_DWORD *)(v15 + 20) & 0xFFFFFFF)));
  else
    v22 = 0;
LABEL_8:
  sub_164B7C0(v21, v15);
  v23 = *(_QWORD *)(v15 + 48);
  v24 = (__int64 *)(v21 + 48);
  v50[0] = v23;
  if ( v23 )
  {
    sub_1623A60((__int64)v50, v23, 2);
    v24 = (__int64 *)(v21 + 48);
    if ( (__int64 *)(v21 + 48) == v50 )
    {
      if ( v50[0] )
        sub_161E7C0((__int64)v50, v50[0]);
      goto LABEL_12;
    }
    v27 = *(_QWORD *)(v21 + 48);
    if ( !v27 )
    {
LABEL_18:
      v28 = (unsigned __int8 *)v50[0];
      *(_QWORD *)(v21 + 48) = v50[0];
      if ( v28 )
        sub_1623210((__int64)v50, v28, (__int64)v24);
      goto LABEL_12;
    }
LABEL_17:
    v46 = v24;
    sub_161E7C0((__int64)v24, v27);
    v24 = v46;
    goto LABEL_18;
  }
  if ( v24 != v50 )
  {
    v27 = *(_QWORD *)(v21 + 48);
    if ( v27 )
      goto LABEL_17;
  }
LABEL_12:
  sub_157F2D0(v22, a1, 0);
  sub_164D160(v15, v21, a3, a4, a5, a6, v25, v26, a9, a10);
  sub_15F20C0((_QWORD *)v15);
  if ( a2 )
    sub_15CDBF0(a2, a1, v22);
}
