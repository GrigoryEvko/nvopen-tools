// Function: sub_13D55B0
// Address: 0x13d55b0
//
__int64 __fastcall sub_13D55B0(_QWORD **a1, __int64 a2)
{
  char v4; // al
  __int64 result; // rax
  __int64 v6; // r13
  unsigned __int64 v7; // rax
  __int64 v8; // rcx
  int v9; // eax
  __int64 *v10; // rax
  __int64 v11; // rax
  _QWORD **v12; // rdi
  __int64 v13; // rax
  __int64 v14; // r13
  unsigned __int64 v15; // rax
  __int64 v16; // rcx
  int v17; // edx
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // r13
  unsigned __int64 v24; // rdx
  __int64 v25; // rsi
  int v26; // edx
  __int64 *v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rax
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rdx
  void *v32; // rcx
  int v33; // eax
  __int64 *v34; // rax
  __int64 v35; // rax
  _QWORD **v36; // rdi
  __int64 v37; // rax
  unsigned __int64 v38; // rax
  void *v39; // rcx
  unsigned __int64 v40; // rdx
  void *v41; // rcx
  __int64 v42; // rcx
  unsigned __int64 v43; // rcx
  void *v44; // rsi

  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 == 51 )
  {
    v6 = *(_QWORD *)(a2 - 48);
    v7 = *(unsigned __int8 *)(v6 + 16);
    if ( (unsigned __int8)v7 <= 0x17u )
    {
      if ( (_BYTE)v7 != 5 )
        goto LABEL_20;
      v40 = *(unsigned __int16 *)(v6 + 18);
      if ( (unsigned __int16)v40 > 0x17u )
        goto LABEL_20;
      v41 = &loc_80A800;
      v9 = (unsigned __int16)v40;
      if ( !_bittest64((const __int64 *)&v41, v40) )
        goto LABEL_20;
    }
    else
    {
      if ( (unsigned __int8)v7 > 0x2Fu )
        goto LABEL_20;
      v8 = 0x80A800000000LL;
      if ( !_bittest64(&v8, v7) )
        goto LABEL_20;
      v9 = (unsigned __int8)v7 - 24;
    }
    if ( v9 == 23 && (*(_BYTE *)(v6 + 17) & 2) != 0 )
    {
      v10 = (*(_BYTE *)(v6 + 23) & 0x40) != 0
          ? *(__int64 **)(v6 - 8)
          : (__int64 *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
      v11 = *v10;
      if ( v11 )
      {
        v12 = a1 + 1;
        **a1 = v11;
        v13 = (*(_BYTE *)(v6 + 23) & 0x40) != 0 ? *(_QWORD *)(v6 - 8) : v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
        result = sub_13D2630(v12, *(_BYTE **)(v13 + 24));
        if ( (_BYTE)result )
        {
          *a1[2] = v6;
          v14 = *(_QWORD *)(a2 - 24);
          if ( v14 )
            goto LABEL_18;
LABEL_21:
          v15 = *(unsigned __int8 *)(v14 + 16);
          if ( (unsigned __int8)v15 <= 0x17u )
          {
            if ( (_BYTE)v15 != 5 )
              return 0;
            v38 = *(unsigned __int16 *)(v14 + 18);
            if ( (unsigned __int16)v38 > 0x17u )
              return 0;
            v39 = &loc_80A800;
            v17 = (unsigned __int16)v38;
            if ( !_bittest64((const __int64 *)&v39, v38) )
              return 0;
          }
          else
          {
            if ( (unsigned __int8)v15 > 0x2Fu )
              return 0;
            v16 = 0x80A800000000LL;
            v17 = (unsigned __int8)v15 - 24;
            if ( !_bittest64(&v16, v15) )
              return 0;
          }
          if ( v17 != 23 || (*(_BYTE *)(v14 + 17) & 2) == 0 )
            return 0;
          v18 = (*(_BYTE *)(v14 + 23) & 0x40) != 0
              ? *(__int64 **)(v14 - 8)
              : (__int64 *)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
          v19 = *v18;
          if ( !v19 )
            return 0;
          **a1 = v19;
          v20 = (*(_BYTE *)(v14 + 23) & 0x40) != 0
              ? *(_QWORD *)(v14 - 8)
              : v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF);
          result = sub_13D2630(a1 + 1, *(_BYTE **)(v20 + 24));
          if ( !(_BYTE)result )
            return 0;
          *a1[2] = v14;
          v21 = *(_QWORD *)(a2 - 48);
          if ( !v21 )
            return 0;
          goto LABEL_33;
        }
      }
    }
LABEL_20:
    v14 = *(_QWORD *)(a2 - 24);
    goto LABEL_21;
  }
  if ( v4 != 5 || *(_WORD *)(a2 + 18) != 27 )
    return 0;
  v22 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v23 = *(_QWORD *)(a2 - 24 * v22);
  v24 = *(unsigned __int8 *)(v23 + 16);
  if ( (unsigned __int8)v24 <= 0x17u )
  {
    if ( (_BYTE)v24 == 5 )
    {
      v43 = *(unsigned __int16 *)(v23 + 18);
      if ( (unsigned __int16)v43 <= 0x17u )
      {
        v44 = &loc_80A800;
        v26 = (unsigned __int16)v43;
        if ( _bittest64((const __int64 *)&v44, v43) )
          goto LABEL_38;
      }
    }
LABEL_46:
    v14 = *(_QWORD *)(a2 + 24 * (1 - v22));
    goto LABEL_47;
  }
  if ( (unsigned __int8)v24 > 0x2Fu )
    goto LABEL_46;
  v25 = 0x80A800000000LL;
  if ( !_bittest64(&v25, v24) )
    goto LABEL_46;
  v26 = (unsigned __int8)v24 - 24;
LABEL_38:
  if ( v26 != 23 || (*(_BYTE *)(v23 + 17) & 2) == 0 )
    goto LABEL_46;
  v27 = (*(_BYTE *)(v23 + 23) & 0x40) != 0
      ? *(__int64 **)(v23 - 8)
      : (__int64 *)(v23 - 24LL * (*(_DWORD *)(v23 + 20) & 0xFFFFFFF));
  v28 = *v27;
  if ( !v28 )
    goto LABEL_46;
  **a1 = v28;
  v29 = sub_13CF970(v23);
  result = sub_13D2630(a1 + 1, *(_BYTE **)(v29 + 24));
  if ( !(_BYTE)result )
  {
    v14 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    goto LABEL_47;
  }
  *a1[2] = v23;
  v14 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  if ( v14 )
  {
LABEL_18:
    *a1[3] = v14;
    return result;
  }
LABEL_47:
  v30 = *(unsigned __int8 *)(v14 + 16);
  if ( (unsigned __int8)v30 <= 0x17u )
  {
    if ( (_BYTE)v30 == 5 )
    {
      v31 = *(unsigned __int16 *)(v14 + 18);
      if ( (unsigned __int16)v31 <= 0x17u )
      {
        v32 = &loc_80A800;
        v33 = (unsigned __int16)v31;
        if ( _bittest64((const __int64 *)&v32, v31) )
          goto LABEL_51;
      }
    }
    return 0;
  }
  if ( (unsigned __int8)v30 > 0x2Fu )
    return 0;
  v42 = 0x80A800000000LL;
  if ( !_bittest64(&v42, v30) )
    return 0;
  v33 = (unsigned __int8)v30 - 24;
LABEL_51:
  if ( v33 != 23 || (*(_BYTE *)(v14 + 17) & 2) == 0 )
    return 0;
  v34 = (*(_BYTE *)(v14 + 23) & 0x40) != 0
      ? *(__int64 **)(v14 - 8)
      : (__int64 *)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
  v35 = *v34;
  if ( !v35 )
    return 0;
  v36 = a1 + 1;
  **a1 = v35;
  v37 = (*(_BYTE *)(v14 + 23) & 0x40) != 0 ? *(_QWORD *)(v14 - 8) : v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF);
  result = sub_13D2630(v36, *(_BYTE **)(v37 + 24));
  if ( !(_BYTE)result )
    return 0;
  *a1[2] = v14;
  v21 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  if ( !v21 )
    return 0;
LABEL_33:
  *a1[3] = v21;
  return result;
}
