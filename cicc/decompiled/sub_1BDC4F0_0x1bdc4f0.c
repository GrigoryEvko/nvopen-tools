// Function: sub_1BDC4F0
// Address: 0x1bdc4f0
//
__int64 __fastcall sub_1BDC4F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  unsigned int v11; // r12d
  __int64 ***v13; // rax
  __int64 **v14; // rbx
  __int64 **v15; // r15
  __int64 *v16; // r8
  double v18; // xmm4_8
  double v19; // xmm5_8
  __int64 v20; // r8
  unsigned int v21; // eax
  __int64 *v22; // rax
  __int64 v23; // r10
  __int64 v24; // r13
  int v25; // eax
  char v26; // dl
  __int64 v27; // rax
  __int64 *v28; // rax
  __int64 v29; // rsi
  __int64 v30; // r13
  int v31; // eax
  __int64 v32; // rax
  __int64 *v33; // rax
  char v34; // al
  char v35; // al
  char v36; // al
  char v37; // al
  __int64 v38; // [rsp+0h] [rbp-40h]
  __int64 v39; // [rsp+0h] [rbp-40h]
  __int64 v40; // [rsp+0h] [rbp-40h]

  if ( !a2 )
    return 0;
  LOBYTE(v11) = (unsigned __int8)(*(_BYTE *)(a2 + 16) - 75) <= 1u
             || (unsigned int)*(unsigned __int8 *)(a2 + 16) - 35 <= 0x11;
  if ( !(_BYTE)v11 )
    return 0;
  v13 = (*(_BYTE *)(a2 + 23) & 0x40) != 0
      ? *(__int64 ****)(a2 - 8)
      : (__int64 ***)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v14 = *v13;
  if ( *((_BYTE *)*v13 + 16) <= 0x17u )
    return 0;
  v15 = v13[3];
  if ( *((_BYTE *)v15 + 16) <= 0x17u )
    return 0;
  v16 = *(__int64 **)(a2 + 40);
  if ( v16 != v14[5] || v16 != v15[5] )
    return 0;
  v38 = *(_QWORD *)(a2 + 40);
  if ( (unsigned __int8)sub_1BDC4B0(a1, *v13, v13[3], a3, a4, a5, a6, a7, a8, a9, a10, a11) )
    return v11;
  v20 = v38;
  v21 = *((unsigned __int8 *)v15 + 16) - 35;
  if ( (unsigned int)*((unsigned __int8 *)v14 + 16) - 35 > 0x11 )
  {
    if ( v21 > 0x11 )
      return 0;
    v33 = v15[1];
    if ( !v33 )
      return 0;
    v14 = (__int64 **)v33[1];
    if ( v14 )
      return 0;
    goto LABEL_17;
  }
  if ( v21 > 0x11 )
  {
    v15 = 0;
    goto LABEL_23;
  }
  v22 = v15[1];
  if ( v22 && !v22[1] )
  {
LABEL_17:
    v23 = (__int64)*(v15 - 6);
    v24 = (__int64)*(v15 - 3);
    v25 = *(unsigned __int8 *)(v23 + 16);
    v26 = *(_BYTE *)(v24 + 16);
    if ( (unsigned __int8)v25 <= 0x17u || (unsigned int)(v25 - 35) > 0x11 )
    {
      if ( (unsigned __int8)(v26 - 35) > 0x11u )
        goto LABEL_22;
    }
    else
    {
      v27 = *(_QWORD *)(v23 + 40);
      if ( (unsigned __int8)(v26 - 35) > 0x11u )
      {
        if ( v38 == v27 )
        {
          v34 = sub_1BDC4B0(a1, v14, (__int64 **)*(v15 - 6), a3, a4, a5, a6, a7, v18, v19, a10, a11);
          v20 = v38;
          if ( v34 )
            return v11;
        }
        goto LABEL_22;
      }
      if ( v38 == v27 )
      {
        v37 = sub_1BDC4B0(a1, v14, (__int64 **)*(v15 - 6), a3, a4, a5, a6, a7, v18, v19, a10, a11);
        v20 = v38;
        if ( v37 )
          return v11;
      }
    }
    if ( v20 == *(_QWORD *)(v24 + 40) )
    {
      v39 = v20;
      v35 = sub_1BDC4B0(a1, v14, (__int64 **)v24, a3, a4, a5, a6, a7, v18, v19, a10, a11);
      v20 = v39;
      if ( v35 )
        return v11;
    }
LABEL_22:
    if ( !v14 )
      return 0;
  }
LABEL_23:
  v28 = v14[1];
  if ( !v28 || v28[1] )
    return 0;
  v29 = (__int64)*(v14 - 6);
  v30 = (__int64)*(v14 - 3);
  v31 = *(unsigned __int8 *)(v29 + 16);
  if ( (unsigned __int8)v31 <= 0x17u || (unsigned int)(v31 - 35) > 0x11 )
  {
    if ( (unsigned __int8)(*(_BYTE *)(v30 + 16) - 35) > 0x11u )
      return 0;
  }
  else
  {
    v32 = *(_QWORD *)(v29 + 40);
    if ( (unsigned __int8)(*(_BYTE *)(v30 + 16) - 35) > 0x11u )
    {
      if ( v20 == v32 && (unsigned __int8)sub_1BDC4B0(a1, (__int64 **)v29, v15, a3, a4, a5, a6, a7, v18, v19, a10, a11) )
        return v11;
      return 0;
    }
    if ( v20 == v32 )
    {
      v40 = v20;
      v36 = sub_1BDC4B0(a1, (__int64 **)v29, v15, a3, a4, a5, a6, a7, v18, v19, a10, a11);
      v20 = v40;
      if ( v36 )
        return v11;
    }
  }
  if ( v20 != *(_QWORD *)(v30 + 40) )
    return 0;
  return sub_1BDC4B0(a1, (__int64 **)v30, v15, a3, a4, a5, a6, a7, v18, v19, a10, a11);
}
