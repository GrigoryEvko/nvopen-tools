// Function: sub_2AC24F0
// Address: 0x2ac24f0
//
void __fastcall sub_2AC24F0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BYTE *v6; // r14
  __int64 v7; // r12
  __int64 v8; // r15
  __int64 v9; // r8
  __int64 v10; // r12
  __int64 v11; // rbx
  __int64 i; // r13
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 *v17; // r13
  __int64 v18; // r12
  __int64 v19; // r8
  __int64 v20; // r9
  char v21; // dl
  __int64 v22; // rax
  _QWORD *v23; // r15
  const void *v24; // rsi
  __int64 *v25; // r14
  __int64 v26; // r13
  _QWORD *v27; // rbx
  _BYTE *v28; // r12
  _QWORD *v29; // rax
  char v30; // dl
  unsigned __int64 v31; // rcx
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r9
  __int64 v35; // [rsp+0h] [rbp-480h]
  __int64 v36; // [rsp+8h] [rbp-478h]
  char v38; // [rsp+20h] [rbp-460h]
  __m128i v39; // [rsp+30h] [rbp-450h] BYREF
  _BYTE *v40; // [rsp+40h] [rbp-440h] BYREF
  __int64 v41; // [rsp+48h] [rbp-438h]
  _BYTE v42[32]; // [rsp+50h] [rbp-430h] BYREF
  __int64 v43; // [rsp+70h] [rbp-410h] BYREF
  char *v44; // [rsp+78h] [rbp-408h]
  __int64 v45; // [rsp+80h] [rbp-400h]
  int v46; // [rsp+88h] [rbp-3F8h]
  char v47; // [rsp+8Ch] [rbp-3F4h]
  char v48; // [rsp+90h] [rbp-3F0h] BYREF
  __int64 v49; // [rsp+B0h] [rbp-3D0h] BYREF
  char *v50; // [rsp+B8h] [rbp-3C8h]
  __int64 v51; // [rsp+C0h] [rbp-3C0h]
  int v52; // [rsp+C8h] [rbp-3B8h]
  char v53; // [rsp+CCh] [rbp-3B4h]
  char v54; // [rsp+D0h] [rbp-3B0h] BYREF
  _QWORD v55[10]; // [rsp+F0h] [rbp-390h] BYREF
  _BYTE v56[344]; // [rsp+140h] [rbp-340h] BYREF
  __int64 v57; // [rsp+298h] [rbp-1E8h]
  _QWORD v58[10]; // [rsp+2A0h] [rbp-1E0h] BYREF
  _BYTE v59[344]; // [rsp+2F0h] [rbp-190h] BYREF
  __int64 v60; // [rsp+448h] [rbp-38h]

  v6 = v42;
  v7 = a1;
  v8 = *(_QWORD *)(a1 + 32);
  v9 = *(_QWORD *)(a1 + 40);
  v40 = v42;
  v41 = 0x400000000LL;
  if ( v9 == v8 )
  {
    LODWORD(v15) = 0;
  }
  else
  {
    v10 = v9;
    do
    {
      v11 = *(_QWORD *)(*(_QWORD *)v8 + 56LL);
      for ( i = *(_QWORD *)v8 + 48LL; i != v11; v11 = *(_QWORD *)(v11 + 8) )
      {
        while ( 1 )
        {
          if ( !v11 )
            BUG();
          if ( *(_BYTE *)(v11 - 24) == 62 && *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v11 - 88) + 8LL) + 8LL) == 2 )
            break;
          v11 = *(_QWORD *)(v11 + 8);
          if ( i == v11 )
            goto LABEL_12;
        }
        v13 = (unsigned int)v41;
        v14 = (unsigned int)v41 + 1LL;
        if ( v14 > HIDWORD(v41) )
        {
          sub_C8D5F0((__int64)&v40, v42, v14, 8u, v9, a6);
          v13 = (unsigned int)v41;
        }
        *(_QWORD *)&v40[8 * v13] = v11 - 24;
        LODWORD(v41) = v41 + 1;
      }
LABEL_12:
      v8 += 8;
    }
    while ( v10 != v8 );
    v7 = a1;
    LODWORD(v15) = v41;
  }
  v16 = v7 + 56;
  v43 = 0;
  v44 = &v48;
  v17 = &v43;
  v45 = 4;
  v46 = 0;
  v47 = 1;
  v49 = 0;
  v50 = &v54;
  v51 = 4;
  v52 = 0;
  v53 = 1;
  if ( !(_DWORD)v15 )
    goto LABEL_35;
  v36 = v7;
  do
  {
    while ( 1 )
    {
      v18 = *(_QWORD *)&v40[8 * (unsigned int)v15 - 8];
      LODWORD(v41) = v15 - 1;
      if ( (unsigned __int8)sub_B19060(v16, *(_QWORD *)(v18 + 40), (__int64)v40, (unsigned int)v15) )
      {
        sub_AE6EC0((__int64)v17, v18);
        if ( v21 )
          break;
      }
      LODWORD(v15) = v41;
LABEL_17:
      if ( !(_DWORD)v15 )
        goto LABEL_31;
    }
    if ( *(_BYTE *)v18 == 75 )
    {
      v29 = sub_AE6EC0((__int64)&v49, v18);
      v31 = (unsigned __int64)(v53 ? &v50[8 * HIDWORD(v51)] : &v50[8 * (unsigned int)v51]);
      v38 = v30;
      v58[0] = v29;
      v58[1] = v31;
      sub_254BBF0((__int64)v58);
      if ( v38 )
      {
        if ( (unsigned __int8)sub_2AA9050(*a2) )
        {
          v35 = **(_QWORD **)(v36 + 32);
          sub_B157E0((__int64)&v39, (_QWORD *)(v18 + 48));
          sub_B17850((__int64)v58, (__int64)"loop-vectorize", (__int64)"VectorMixedPrecision", 20, &v39, v35);
          sub_B18290((__int64)v58, "floating point conversion changes vector width. ", 0x30u);
          sub_B18290((__int64)v58, "Mixed floating point precision requires an up/down ", 0x33u);
          sub_B18290((__int64)v58, "cast that will negatively impact performance.", 0x2Du);
          sub_23FE290((__int64)v55, (__int64)v58, v32, v33, (__int64)v55, v34);
          v57 = v60;
          v55[0] = &unk_49D9DE8;
          v58[0] = &unk_49D9D40;
          sub_23FD590((__int64)v59);
          sub_1049740(a2, (__int64)v55);
          v55[0] = &unk_49D9D40;
          sub_23FD590((__int64)v56);
        }
      }
    }
    v22 = 4LL * (*(_DWORD *)(v18 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(v18 + 7) & 0x40) != 0 )
    {
      v23 = *(_QWORD **)(v18 - 8);
      v18 = (__int64)&v23[v22];
    }
    else
    {
      v23 = (_QWORD *)(v18 - v22 * 8);
    }
    v15 = (unsigned int)v41;
    if ( (_QWORD *)v18 == v23 )
      goto LABEL_17;
    v24 = v6;
    v25 = v17;
    v26 = v16;
    v27 = (_QWORD *)v18;
    do
    {
      v28 = (_BYTE *)*v23;
      if ( *(_BYTE *)*v23 > 0x1Cu )
      {
        if ( v15 + 1 > (unsigned __int64)HIDWORD(v41) )
        {
          sub_C8D5F0((__int64)&v40, v24, v15 + 1, 8u, v19, v20);
          v15 = (unsigned int)v41;
        }
        *(_QWORD *)&v40[8 * v15] = v28;
        v15 = (unsigned int)(v41 + 1);
        LODWORD(v41) = v41 + 1;
      }
      v23 += 4;
    }
    while ( v27 != v23 );
    v16 = v26;
    v17 = v25;
    v6 = v24;
  }
  while ( (_DWORD)v15 );
LABEL_31:
  if ( !v53 )
    _libc_free((unsigned __int64)v50);
  if ( !v47 )
    _libc_free((unsigned __int64)v44);
LABEL_35:
  if ( v40 != v6 )
    _libc_free((unsigned __int64)v40);
}
