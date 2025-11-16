// Function: sub_2CB1240
// Address: 0x2cb1240
//
_BYTE *__fastcall sub_2CB1240(__int64 a1, _BYTE *a2, __int64 *a3)
{
  _BYTE *v3; // r12
  char v4; // al
  _BYTE *v7; // rcx
  __int64 v8; // rdx
  unsigned int v9; // edi
  __int64 v10; // rsi
  unsigned __int64 v11; // rdx
  _QWORD *v12; // r12
  __int64 v14; // rax
  __int64 v15; // rax
  _BYTE *v16; // rdx
  __int64 v17; // rdx
  __int64 *v18; // rsi
  unsigned int v19; // edx
  __int64 v20; // rax
  _QWORD *v21; // r13
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r13
  int v25; // eax
  unsigned int *v26; // rcx
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  __int64 v29; // rax
  unsigned int *v30; // rcx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  unsigned int *v34; // [rsp+8h] [rbp-68h]
  _BYTE *v35; // [rsp+10h] [rbp-60h]
  unsigned int *v36; // [rsp+20h] [rbp-50h]
  unsigned int *v37; // [rsp+28h] [rbp-48h]
  unsigned int v38; // [rsp+28h] [rbp-48h]
  unsigned __int64 v39; // [rsp+30h] [rbp-40h] BYREF
  __int64 v40; // [rsp+38h] [rbp-38h]

  v3 = a2;
  v4 = *a2;
  if ( *a2 <= 0x1Cu )
    return v3;
  while ( 1 )
  {
    if ( (unsigned __int8)(v4 - 68) <= 1u )
      goto LABEL_13;
    if ( v4 == 54 )
    {
      if ( (v3[7] & 0x40) != 0 )
        v16 = (_BYTE *)*((_QWORD *)v3 - 1);
      else
        v16 = &v3[-32 * (*((_DWORD *)v3 + 1) & 0x7FFFFFF)];
      v17 = *((_QWORD *)v16 + 4);
      if ( *(_BYTE *)v17 != 17 )
        return v3;
      v18 = *(__int64 **)(v17 + 24);
      v19 = *(_DWORD *)(v17 + 32);
      if ( v19 > 0x40 )
      {
        v14 = *v18 + 7;
        if ( *v18 >= 0 )
          v14 = *v18;
        v15 = v14 >> 3;
      }
      else
      {
        v15 = 0;
        if ( v19 )
          v15 = ((__int64)((_QWORD)v18 << (64 - (unsigned __int8)v19)) >> (64 - (unsigned __int8)v19)) / 8;
      }
      if ( *a3 < v15 )
        return v3;
      *a3 -= v15;
      if ( (v3[7] & 0x40) == 0 )
      {
LABEL_14:
        v12 = &v3[-32 * (*((_DWORD *)v3 + 1) & 0x7FFFFFF)];
LABEL_15:
        v3 = (_BYTE *)*v12;
        goto LABEL_16;
      }
LABEL_23:
      v12 = (_QWORD *)*((_QWORD *)v3 - 1);
      goto LABEL_15;
    }
    if ( (unsigned __int8)(v4 - 55) <= 1u )
    {
      if ( (v3[7] & 0x40) != 0 )
        v7 = (_BYTE *)*((_QWORD *)v3 - 1);
      else
        v7 = &v3[-32 * (*((_DWORD *)v3 + 1) & 0x7FFFFFF)];
      v8 = *((_QWORD *)v7 + 4);
      if ( *(_BYTE *)v8 != 17 )
        return v3;
      v9 = *(_DWORD *)(v8 + 32);
      v10 = *a3;
      v11 = *(_QWORD *)(v8 + 24);
      if ( v4 == 56 )
      {
        if ( v9 > 0x40 )
        {
          v10 += *(_QWORD *)v11 / 8LL;
        }
        else if ( v9 )
        {
          v10 = ((__int64)(v11 << (64 - (unsigned __int8)v9)) >> (64 - (unsigned __int8)v9)) / 8 + *a3;
        }
      }
      else
      {
        if ( v9 > 0x40 )
          v11 = *(_QWORD *)v11;
        v10 += v11 >> 3;
      }
      *a3 = v10;
LABEL_13:
      if ( (v3[7] & 0x40) == 0 )
        goto LABEL_14;
      goto LABEL_23;
    }
    if ( v4 == 93 )
      break;
    if ( v4 != 90 )
      return v3;
    v20 = *((_QWORD *)v3 - 4);
    v21 = *(_QWORD **)(v20 + 24);
    if ( *(_DWORD *)(v20 + 32) > 0x40u )
      v21 = (_QWORD *)*v21;
    v22 = sub_9208B0(a1, *(_QWORD *)(*(_QWORD *)(*((_QWORD *)v3 - 8) + 8LL) + 24LL));
    v40 = v23;
    v39 = (unsigned int)v21 * ((unsigned __int64)(v22 + 7) >> 3);
    *a3 += sub_CA1930(&v39);
    v3 = (_BYTE *)*((_QWORD *)v3 - 8);
LABEL_16:
    v4 = *v3;
    if ( *v3 <= 0x1Cu )
      return v3;
  }
  v24 = *(_QWORD *)(*((_QWORD *)v3 - 4) + 8LL);
  v35 = (_BYTE *)*((_QWORD *)v3 - 4);
  v25 = *(unsigned __int8 *)(v24 + 8);
  if ( (unsigned int)(v25 - 15) > 1 )
    return v3;
  v26 = (unsigned int *)*((_QWORD *)v3 + 9);
  v36 = &v26[*((unsigned int *)v3 + 20)];
  if ( v26 == v36 )
  {
LABEL_49:
    v3 = v35;
    goto LABEL_16;
  }
  if ( (_BYTE)v25 == 15 )
  {
LABEL_40:
    v37 = v26;
    v27 = 16LL * *v26 + sub_AE4AC0(a1, v24) + 24;
    v28 = *(_QWORD *)v27;
    LOBYTE(v27) = *(_BYTE *)(v27 + 8);
    v39 = v28;
    LOBYTE(v40) = v27;
    v29 = sub_CA1930(&v39);
    v30 = v37;
    *a3 += v29;
    v24 = *(_QWORD *)(*(_QWORD *)(v24 + 16) + 8LL * *v37);
    goto LABEL_41;
  }
  while ( (_BYTE)v25 == 16 )
  {
    v24 = *(_QWORD *)(v24 + 24);
    v34 = v26;
    v38 = *v26;
    v31 = sub_9208B0(a1, v24);
    v40 = v32;
    v39 = v38 * ((unsigned __int64)(v31 + 7) >> 3);
    v33 = sub_CA1930(&v39);
    v30 = v34;
    *a3 += v33;
LABEL_41:
    v26 = v30 + 1;
    if ( v26 == v36 )
      goto LABEL_49;
    LOBYTE(v25) = *(_BYTE *)(v24 + 8);
    if ( (_BYTE)v25 == 15 )
      goto LABEL_40;
  }
  return v3;
}
