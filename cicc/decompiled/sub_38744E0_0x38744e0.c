// Function: sub_38744E0
// Address: 0x38744e0
//
__int64 ***__fastcall sub_38744E0(
        __int64 *a1,
        __int64 a2,
        __int64 **a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  int v12; // eax
  double v13; // xmm4_8
  double v14; // xmm5_8
  int v15; // r13d
  unsigned __int8 v16; // al
  __int64 ***result; // rax
  __int64 i; // r8
  int v19; // ecx
  __int64 v20; // rax
  __int64 v21; // rbx
  bool v22; // zf
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // rax
  char v26; // al
  __int64 v27; // rax
  __int64 v28; // rbx

  v12 = sub_15FBEB0((_QWORD *)a2, 0, (__int64)a3, 0);
  if ( v12 == 48 )
  {
    v20 = sub_386F050(a2, a1[34]);
    v19 = 48;
    i = v20;
    return (__int64 ***)sub_38742E0((__int64)a1, a2, (__int64)a3, v19, i, a4, a5, a6, a7, v13, v14, a10, a11);
  }
  v15 = v12;
  if ( v12 != 47 )
  {
    if ( (unsigned int)(v12 - 45) > 1 )
    {
LABEL_4:
      v16 = *(_BYTE *)(a2 + 16);
LABEL_5:
      if ( v16 <= 0x10u )
        return (__int64 ***)sub_15A46C0(v15, (__int64 ***)a2, a3, 0);
      if ( v16 == 17 )
      {
        v24 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 80LL);
        if ( !v24 )
          BUG();
        for ( i = *(_QWORD *)(v24 + 24); ; i = *(_QWORD *)(i + 8) )
        {
          if ( !i )
            BUG();
          v26 = *(_BYTE *)(i - 8);
          if ( v26 == 71 )
          {
            v27 = *(_QWORD *)(i - 48);
            if ( !v27 )
              BUG();
            if ( *(_BYTE *)(v27 + 16) != 17 || a2 == v27 )
              goto LABEL_11;
          }
          else
          {
            if ( v26 != 78 )
              goto LABEL_11;
            v25 = *(_QWORD *)(i - 48);
            if ( *(_BYTE *)(v25 + 16)
              || (*(_BYTE *)(v25 + 33) & 0x20) == 0
              || (unsigned int)(*(_DWORD *)(v25 + 36) - 35) > 3 )
            {
              goto LABEL_11;
            }
          }
        }
      }
      goto LABEL_10;
    }
    v21 = sub_1456C90(*a1, (__int64)a3);
    v22 = v21 == sub_1456C90(*a1, *(_QWORD *)a2);
    v16 = *(_BYTE *)(a2 + 16);
    if ( !v22 )
      goto LABEL_5;
    if ( v16 <= 0x17u )
    {
LABEL_16:
      if ( v16 != 5 )
        goto LABEL_5;
      if ( (unsigned int)*(unsigned __int16 *)(a2 + 18) - 45 > 1 )
        return (__int64 ***)sub_15A46C0(v15, (__int64 ***)a2, a3, 0);
      v23 = sub_1456C90(*a1, *(_QWORD *)a2);
      if ( v23 != sub_1456C90(*a1, **(_QWORD **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) )
        goto LABEL_4;
      return *(__int64 ****)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    }
    if ( (unsigned __int8)(v16 - 69) <= 1u )
    {
      v28 = sub_1456C90(*a1, *(_QWORD *)a2);
      if ( v28 == sub_1456C90(*a1, **(_QWORD **)(a2 - 24)) )
        return *(__int64 ****)(a2 - 24);
      v16 = *(_BYTE *)(a2 + 16);
      goto LABEL_16;
    }
LABEL_10:
    i = sub_386F050(a2, a1[34]);
LABEL_11:
    v19 = v15;
    return (__int64 ***)sub_38742E0((__int64)a1, a2, (__int64)a3, v19, i, a4, a5, a6, a7, v13, v14, a10, a11);
  }
  if ( a3 == *(__int64 ***)a2 )
    return (__int64 ***)a2;
  v16 = *(_BYTE *)(a2 + 16);
  if ( (unsigned __int8)(v16 - 60) > 0xCu )
    goto LABEL_5;
  result = *(__int64 ****)(a2 - 24);
  if ( a3 != *result )
    goto LABEL_10;
  return result;
}
