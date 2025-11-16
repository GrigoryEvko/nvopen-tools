// Function: sub_281E0D0
// Address: 0x281e0d0
//
__int64 __fastcall sub_281E0D0(
        __int64 a1,
        unsigned __int8 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        _QWORD *a6,
        __int64 a7)
{
  __int64 v7; // r12
  __int64 v10; // r15
  int v11; // eax
  char v12; // cl
  __int64 v13; // r14
  unsigned int v14; // r15d
  __int64 v15; // r15
  unsigned __int8 *v16; // r13
  unsigned __int8 **v17; // rax
  unsigned __int8 **v18; // rdx
  __int64 *v20; // rax
  __int64 *v21; // rax
  unsigned __int8 v22; // r13
  __int64 v23; // r12
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rax
  __int64 v26; // [rsp+8h] [rbp-348h]
  __int64 v27; // [rsp+10h] [rbp-340h]
  __int64 v28; // [rsp+28h] [rbp-328h]
  __int64 v29; // [rsp+28h] [rbp-328h]
  int v31; // [rsp+38h] [rbp-318h]
  __int64 v32; // [rsp+38h] [rbp-318h]
  char v33; // [rsp+38h] [rbp-318h]
  __m128i v34; // [rsp+40h] [rbp-310h] BYREF
  __int64 v35; // [rsp+50h] [rbp-300h]
  __int64 v36; // [rsp+58h] [rbp-2F8h]
  __int64 v37; // [rsp+60h] [rbp-2F0h]
  __int64 v38; // [rsp+68h] [rbp-2E8h]
  char v39; // [rsp+70h] [rbp-2E0h]
  _QWORD v40[2]; // [rsp+80h] [rbp-2D0h] BYREF
  __int64 v41; // [rsp+90h] [rbp-2C0h]
  __int64 v42; // [rsp+98h] [rbp-2B8h] BYREF
  unsigned int v43; // [rsp+A0h] [rbp-2B0h]
  _QWORD v44[2]; // [rsp+1D8h] [rbp-178h] BYREF
  char v45; // [rsp+1E8h] [rbp-168h]
  _BYTE *v46; // [rsp+1F0h] [rbp-160h]
  __int64 v47; // [rsp+1F8h] [rbp-158h]
  _BYTE v48[128]; // [rsp+200h] [rbp-150h] BYREF
  __int16 v49; // [rsp+280h] [rbp-D0h]
  _QWORD v50[2]; // [rsp+288h] [rbp-C8h] BYREF
  __int64 v51; // [rsp+298h] [rbp-B8h]
  __int64 v52; // [rsp+2A0h] [rbp-B0h] BYREF
  unsigned int v53; // [rsp+2A8h] [rbp-A8h]
  char v54; // [rsp+320h] [rbp-30h] BYREF

  v28 = 0xBFFFFFFFFFFFFFFELL;
  if ( *(_WORD *)(a4 + 24) || *(_WORD *)(a5 + 24) )
    goto LABEL_11;
  v10 = *(_QWORD *)(a4 + 32);
  if ( *(_DWORD *)(v10 + 32) <= 0x40u )
  {
    v7 = *(_QWORD *)(v10 + 24);
    goto LABEL_6;
  }
  v31 = *(_DWORD *)(v10 + 32);
  v11 = sub_C444A0(v10 + 24);
  v12 = 0;
  if ( (unsigned int)(v31 - v11) <= 0x40 )
  {
    v7 = **(_QWORD **)(v10 + 24);
LABEL_6:
    v12 = 1;
  }
  v13 = *(_QWORD *)(a5 + 32);
  v14 = *(_DWORD *)(v13 + 32);
  if ( v14 <= 0x40 )
  {
    v29 = *(_QWORD *)(v13 + 24);
    if ( !v29 )
    {
      v25 = 0xBFFFFFFFFFFFFFFELL;
      if ( v12 )
        v25 = 0;
      v28 = v25;
      goto LABEL_11;
    }
    goto LABEL_9;
  }
  v33 = v12;
  v28 = 0xBFFFFFFFFFFFFFFELL;
  if ( v14 - (unsigned int)sub_C444A0(v13 + 24) <= 0x40 )
  {
    v12 = v33;
    v29 = **(_QWORD **)(v13 + 24);
LABEL_9:
    if ( v12 )
    {
      v23 = v7 + 1;
      v24 = 0xBFFFFFFFFFFFFFFELL;
      if ( (unsigned __int64)(v23 * v29) <= 0x3FFFFFFFFFFFFFFBLL )
        v24 = v23 * v29;
      v28 = v24;
    }
    else
    {
      v28 = 0xBFFFFFFFFFFFFFFELL;
    }
  }
LABEL_11:
  v26 = *(_QWORD *)(a3 + 40);
  v27 = *(_QWORD *)(a3 + 32);
  if ( v27 == v26 )
    return 0;
  while ( 1 )
  {
    v15 = *(_QWORD *)(*(_QWORD *)v27 + 56LL);
    if ( *(_QWORD *)v27 + 48LL != v15 )
      break;
LABEL_22:
    v27 += 8;
    if ( v26 == v27 )
      return 0;
  }
  v32 = *(_QWORD *)v27 + 48LL;
  while ( 1 )
  {
    v16 = (unsigned __int8 *)(v15 - 24);
    if ( !v15 )
      v16 = 0;
    if ( *(_BYTE *)(a7 + 28) )
      break;
    if ( !sub_C8CA60(a7, (__int64)v16) )
      goto LABEL_25;
LABEL_21:
    v15 = *(_QWORD *)(v15 + 8);
    if ( v32 == v15 )
      goto LABEL_22;
  }
  v17 = *(unsigned __int8 ***)(a7 + 8);
  v18 = &v17[*(unsigned int *)(a7 + 20)];
  if ( v17 != v18 )
  {
    while ( v16 != *v17 )
    {
      if ( v18 == ++v17 )
        goto LABEL_25;
    }
    goto LABEL_21;
  }
LABEL_25:
  v35 = 0;
  v36 = 0;
  v34.m128i_i64[0] = a1;
  v37 = 0;
  v34.m128i_i64[1] = v28;
  v38 = 0;
  v39 = 1;
  v40[1] = 0;
  v41 = 1;
  v40[0] = a6;
  v20 = &v42;
  do
  {
    *v20 = -4;
    v20 += 5;
    *(v20 - 4) = -3;
    *(v20 - 3) = -4;
    *(v20 - 2) = -3;
  }
  while ( v20 != v44 );
  v44[1] = 0;
  v47 = 0x400000000LL;
  v49 = 256;
  v44[0] = v50;
  v45 = 0;
  v46 = v48;
  v50[1] = 0;
  v51 = 1;
  v50[0] = &unk_49DDBE8;
  v21 = &v52;
  do
  {
    *v21 = -4096;
    v21 += 2;
  }
  while ( v21 != (__int64 *)&v54 );
  v22 = sub_CF63E0(a6, v16, &v34, (__int64)v40);
  v50[0] = &unk_49DDBE8;
  if ( (v51 & 1) == 0 )
    sub_C7D6A0(v52, 16LL * v53, 8);
  nullsub_184();
  if ( v46 != v48 )
    _libc_free((unsigned __int64)v46);
  if ( (v41 & 1) == 0 )
    sub_C7D6A0(v42, 40LL * v43, 8);
  if ( (v22 & a2) == 0 )
    goto LABEL_21;
  return 1;
}
