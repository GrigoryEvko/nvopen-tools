// Function: sub_1780430
// Address: 0x1780430
//
_QWORD *__fastcall sub_1780430(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  unsigned __int64 v6; // rbx
  __int64 v7; // r13
  __int64 **v8; // r15
  unsigned __int8 v9; // al
  unsigned int v10; // r14d
  __int64 v11; // rdx
  __int64 v12; // rdx
  _QWORD *v13; // r12
  int v15; // edx
  __int64 ****v16; // rdx
  __int64 ***v17; // r11
  int v18; // edx
  int v19; // edx
  __int64 ****v20; // rcx
  __int64 ***v21; // rcx
  __int64 v22; // rsi
  __int64 v23; // r14
  _QWORD *v24; // rax
  int v25; // edx
  __int64 ****v26; // rdx
  int v27; // edx
  int v28; // edx
  __int64 ****v29; // rdx
  __int64 v30; // rax
  bool v31; // cc
  __int64 ***v32; // [rsp+8h] [rbp-68h]
  __int64 v33; // [rsp+10h] [rbp-60h]
  __int64 v34; // [rsp+18h] [rbp-58h]
  __int64 v35[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v36; // [rsp+30h] [rbp-40h]

  v6 = *(_QWORD *)(a1 - 48);
  v7 = *(_QWORD *)(a1 - 24);
  v8 = *(__int64 ***)a1;
  v9 = *(_BYTE *)(v6 + 16);
  v10 = *(unsigned __int8 *)(a1 + 16) - 24;
  if ( v9 > 0x17u )
  {
    v15 = v9 - 24;
  }
  else
  {
    if ( v9 != 5 )
      goto LABEL_3;
    v15 = *(unsigned __int16 *)(v6 + 18);
  }
  if ( v15 != 37 )
    goto LABEL_3;
  v16 = (*(_BYTE *)(v6 + 23) & 0x40) != 0
      ? *(__int64 *****)(v6 - 8)
      : (__int64 ****)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
  v17 = *v16;
  if ( !*v16 )
    goto LABEL_3;
  v18 = *(unsigned __int8 *)(v7 + 16);
  if ( (unsigned __int8)v18 > 0x17u )
  {
    v19 = v18 - 24;
  }
  else
  {
    if ( (_BYTE)v18 != 5 )
      goto LABEL_3;
    v19 = *(unsigned __int16 *)(v7 + 18);
  }
  if ( v19 != 37
    || ((*(_BYTE *)(v7 + 23) & 0x40) == 0
      ? (v20 = (__int64 ****)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF)))
      : (v20 = *(__int64 *****)(v7 - 8)),
        (v21 = *v20) == 0 || *v21 != *v17) )
  {
LABEL_3:
    v11 = *(_QWORD *)(v6 + 8);
    if ( !v11 || *(_QWORD *)(v11 + 8) )
    {
LABEL_5:
      v12 = *(_QWORD *)(v7 + 8);
      if ( !v12 )
        return 0;
      goto LABEL_6;
    }
LABEL_32:
    if ( v9 <= 0x17u )
    {
      if ( v9 != 5 )
        goto LABEL_5;
      v25 = *(unsigned __int16 *)(v6 + 18);
    }
    else
    {
      v25 = v9 - 24;
    }
    if ( v25 == 37 )
    {
      v26 = (*(_BYTE *)(v6 + 23) & 0x40) != 0
          ? *(__int64 *****)(v6 - 8)
          : (__int64 ****)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
      v32 = *v26;
      if ( *v26 )
      {
        if ( *(_BYTE *)(v7 + 16) <= 0x10u )
        {
          v6 = *(_QWORD *)(a1 - 24);
          goto LABEL_54;
        }
      }
    }
    goto LABEL_5;
  }
  v22 = *(_QWORD *)(v6 + 8);
  if ( v22 )
  {
    if ( !*(_QWORD *)(v22 + 8) )
    {
LABEL_24:
      v36 = 257;
LABEL_25:
      v23 = sub_17066B0(a2, v10, (__int64)v17, (__int64)v21, v35, 0, a3, a4, a5);
      goto LABEL_26;
    }
    v12 = *(_QWORD *)(v7 + 8);
    if ( !v12 )
    {
      if ( *(_QWORD *)(v22 + 8) )
        return 0;
      goto LABEL_32;
    }
  }
  else
  {
    v12 = *(_QWORD *)(v7 + 8);
    if ( !v12 )
      return 0;
  }
  if ( !*(_QWORD *)(v12 + 8) )
    goto LABEL_24;
  if ( !v22 )
    return 0;
  if ( !*(_QWORD *)(v22 + 8) )
    goto LABEL_32;
LABEL_6:
  if ( *(_QWORD *)(v12 + 8) )
    return 0;
  v27 = *(unsigned __int8 *)(v7 + 16);
  if ( (unsigned __int8)v27 > 0x17u )
  {
    v28 = v27 - 24;
  }
  else
  {
    if ( (_BYTE)v27 != 5 )
      return 0;
    v28 = *(unsigned __int16 *)(v7 + 18);
  }
  if ( v28 != 37 )
    return 0;
  v29 = (*(_BYTE *)(v7 + 23) & 0x40) != 0
      ? *(__int64 *****)(v7 - 8)
      : (__int64 ****)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF));
  v32 = *v29;
  if ( !*v29 || v9 > 0x10u )
    return 0;
LABEL_54:
  v33 = a2;
  v13 = 0;
  v34 = sub_15A43B0(v6, *v32, 0);
  v30 = sub_15A3CB0(v34, v8, 0);
  v17 = (__int64 ***)v34;
  a2 = v33;
  if ( v6 != v30 )
    return v13;
  v31 = *(_BYTE *)(v7 + 16) <= 0x10u;
  v36 = 257;
  if ( !v31 )
  {
    v21 = v32;
    goto LABEL_25;
  }
  v23 = sub_17066B0(v33, v10, (__int64)v32, v34, v35, 0, a3, a4, a5);
LABEL_26:
  v36 = 257;
  v24 = sub_1648A60(56, 1u);
  v13 = v24;
  if ( v24 )
    sub_15FC690((__int64)v24, v23, (__int64)v8, (__int64)v35, 0);
  return v13;
}
