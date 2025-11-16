// Function: sub_1783D50
// Address: 0x1783d50
//
__int64 __fastcall sub_1783D50(
        __int64 *a1,
        _BYTE *a2,
        __int64 a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  _BYTE *v12; // rbx
  __int64 v13; // r14
  __int64 v14; // r15
  __int64 v15; // rax
  unsigned __int8 *v16; // rax
  __int64 v17; // rcx
  unsigned __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v21; // rcx
  double v22; // xmm4_8
  double v23; // xmm5_8
  __int64 v24; // rdx
  unsigned __int8 v25; // al
  __int64 v26; // rax
  unsigned int v27; // ecx
  _QWORD *v28; // r14
  int v29; // eax
  unsigned int v30; // ecx
  __int64 v31; // rax
  unsigned int v32; // [rsp+Ch] [rbp-34h]
  unsigned int v33; // [rsp+Ch] [rbp-34h]

  v12 = a2;
  v13 = *((_QWORD *)a2 - 3);
  v14 = *((_QWORD *)a2 - 6);
  v15 = *(_QWORD *)(v13 + 8);
  if ( !v15
    || *(_QWORD *)(v15 + 8)
    || (v16 = sub_1780EA0(v13, (__int64)a1, (__int64)a2, a4, *(double *)a5.m128_u64, a6, a7)) == 0 )
  {
    if ( (unsigned __int8)sub_1781E40(a1, (__int64)a2, a3, a4) )
      return (__int64)v12;
    v24 = *(unsigned __int8 *)(v13 + 16);
    if ( (unsigned __int8)v24 > 0x10u )
      return 0;
    v25 = *(_BYTE *)(v14 + 16);
    if ( v25 <= 0x17u )
      return 0;
    if ( v25 == 79 )
    {
      v26 = sub_1707470((__int64)a1, a2, v14, *(double *)a5.m128_u64, a6, a7);
      if ( v26 )
        return v26;
      goto LABEL_18;
    }
    if ( v25 != 77
      || (_BYTE)v24 != 13
      && (*(_BYTE *)(*(_QWORD *)v13 + 8LL) != 16
       || (v31 = sub_15A1020((_BYTE *)v13, (__int64)a2, v24, v21), (v13 = v31) == 0)
       || *(_BYTE *)(v31 + 16) != 13) )
    {
LABEL_18:
      if ( (unsigned __int8)sub_17AD890(a1, a2) )
        return (__int64)v12;
      return 0;
    }
    v27 = *(_DWORD *)(v13 + 32);
    v28 = (_QWORD *)(v13 + 24);
    if ( v27 <= 0x40 )
    {
      if ( !*v28 || a2[16] != 44 && *v28 == 1LL << ((unsigned __int8)v27 - 1) )
        goto LABEL_18;
    }
    else
    {
      v32 = v27;
      v29 = sub_16A57B0((__int64)v28);
      v30 = v32;
      if ( v32 == v29 )
        goto LABEL_18;
      if ( a2[16] != 44 )
      {
        v33 = v32 - 1;
        if ( (*(_QWORD *)(*v28 + 8LL * ((v30 - 1) >> 6)) & (1LL << ((unsigned __int8)v30 - 1))) != 0
          && v33 == (unsigned int)sub_16A58A0((__int64)v28) )
        {
          goto LABEL_18;
        }
      }
    }
    v26 = sub_17127D0(a1, (__int64)a2, v14, a5, a6, a7, a8, v22, v23, a11, a12);
    if ( v26 )
      return v26;
    goto LABEL_18;
  }
  if ( *((_QWORD *)a2 - 3) )
  {
    v17 = *((_QWORD *)a2 - 2);
    v18 = *((_QWORD *)a2 - 1) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v18 = v17;
    if ( v17 )
      *(_QWORD *)(v17 + 16) = *(_QWORD *)(v17 + 16) & 3LL | v18;
  }
  *((_QWORD *)a2 - 3) = v16;
  v19 = *((_QWORD *)v16 + 1);
  *((_QWORD *)a2 - 2) = v19;
  if ( v19 )
    *(_QWORD *)(v19 + 16) = (unsigned __int64)(a2 - 16) | *(_QWORD *)(v19 + 16) & 3LL;
  *((_QWORD *)a2 - 1) = *((_QWORD *)a2 - 1) & 3LL | (unsigned __int64)(v16 + 8);
  *((_QWORD *)v16 + 1) = a2 - 24;
  return (__int64)v12;
}
