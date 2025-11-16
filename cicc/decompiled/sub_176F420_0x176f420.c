// Function: sub_176F420
// Address: 0x176f420
//
__int64 __fastcall sub_176F420(
        __int64 *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10,
        __int64 a11,
        __int64 a12)
{
  __int64 *v13; // rdi
  unsigned __int8 v14; // al
  __int64 *v15; // r13
  __int64 v16; // r15
  unsigned __int8 v17; // al
  __int64 result; // rax
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rax

  v13 = *(__int64 **)(a2 - 24);
  v14 = *((_BYTE *)v13 + 16);
  v15 = v13 + 3;
  if ( v14 != 13 )
  {
    if ( *(_BYTE *)(*v13 + 8) != 16 )
      return 0;
    if ( v14 > 0x10u )
      return 0;
    v19 = sub_15A1020(v13, a2, *v13, a12);
    if ( !v19 )
      return 0;
    v15 = (__int64 *)(v19 + 24);
    if ( *(_BYTE *)(v19 + 16) != 13 )
      return 0;
  }
  v16 = *(_QWORD *)(a2 - 48);
  v17 = *(_BYTE *)(v16 + 16);
  if ( v17 > 0x17u )
  {
    switch ( v17 )
    {
      case '#':
        result = (__int64)sub_17594C0((__int64)a1, a2, *(__int64 **)(a2 - 48), v15, *(double *)a3.m128_u64, a4, a5);
        if ( !result )
          goto LABEL_6;
        return result;
      case '$':
      case '&':
      case '(':
      case '+':
      case ',':
      case '-':
      case '.':
        goto LABEL_6;
      case '%':
        result = (__int64)sub_175A220(
                            (__int64)a1,
                            a2,
                            *(_QWORD *)(a2 - 48),
                            (__int64)v15,
                            *(double *)a3.m128_u64,
                            a4,
                            a5);
        if ( !result )
          goto LABEL_6;
        return result;
      case '\'':
        result = (__int64)sub_1758A30((__int64)a1, a2, *(__int64 ****)(a2 - 48), (__int64)v15);
        if ( !result )
          goto LABEL_6;
        return result;
      case ')':
        result = (__int64)sub_1758BB0((__int64)a1, a2, *(_QWORD *)(a2 - 48), (__int64)v15);
        if ( !result )
          goto LABEL_5;
        return result;
      case '*':
LABEL_5:
        result = sub_175E1D0((__int64)a1, a2, v16, (__int64)v15, a3, a4, a5, a6, a7, a8, a9, a10);
        if ( !result )
          goto LABEL_6;
        return result;
      case '/':
        result = (__int64)sub_1767F00(a1, a2, *(_QWORD *)(a2 - 48), (__int64)v15, a3, a4, a5, a6, a7, a8, a9, a10);
        if ( !result )
          goto LABEL_6;
        return result;
      case '0':
      case '1':
        result = sub_1761F10(a1, a2, *(_QWORD *)(a2 - 48), (__int64)v15, a3, a4, a5, a6, a7, a8, a9, a10);
        if ( !result )
          goto LABEL_6;
        return result;
      case '2':
        result = sub_1766760(a1, (_WORD *)a2, *(_QWORD *)(a2 - 48), (__int64)v15, a3, a4, a5, a6, a7, a8, a9, a10);
        if ( result )
          return result;
        result = (__int64)sub_17612E0((__int64)a1, a2, (_QWORD *)v16, (__int64)v15, *(double *)a3.m128_u64, a4, a5);
        if ( result )
          return result;
        goto LABEL_6;
      case '3':
        result = sub_176E680((__int64)a1, a2, *(_QWORD *)(a2 - 48), (__int64)v15);
        if ( !result )
          goto LABEL_6;
        return result;
      case '4':
        result = (__int64)sub_1765920(a1, a2, *(_QWORD *)(a2 - 48), (__int64)v15);
        if ( result )
          return result;
LABEL_6:
        result = (__int64)sub_175AA80((__int64)a1, a2, v16, (__int64)v15, *(double *)a3.m128_u64, a4, a5);
        if ( result )
          return result;
        v16 = *(_QWORD *)(a2 - 48);
        v17 = *(_BYTE *)(v16 + 16);
LABEL_8:
        if ( v17 != 79 )
          break;
        v20 = *(_QWORD *)(a2 - 24);
        if ( *(_BYTE *)(v20 + 16) != 13 )
          return sub_1765500(a1, a2, (_QWORD **)v15);
        v21 = *(_QWORD *)(a2 + 8);
        if ( !v21 || *(_QWORD *)(v21 + 8) )
          return sub_1765500(a1, a2, (_QWORD **)v15);
        result = sub_1767840(a1, a2, v16, (_QWORD *)v20, a3, a4, a5, a6, a7, a8, a9, a10);
        if ( result )
          return result;
        v16 = *(_QWORD *)(a2 - 48);
        v17 = *(_BYTE *)(v16 + 16);
        break;
      default:
        goto LABEL_8;
    }
  }
  if ( v17 == 60 )
  {
    result = (__int64)sub_176EDE0(a1, a2, (__int64 *)v16, (__int64)v15);
    if ( result )
      return result;
    v16 = *(_QWORD *)(a2 - 48);
    v17 = *(_BYTE *)(v16 + 16);
  }
  if ( v17 != 71 )
    return sub_1765500(a1, a2, (_QWORD **)v15);
  result = (__int64)sub_1758F20((__int64)a1, a2, v16, v15);
  if ( !result )
    return sub_1765500(a1, a2, (_QWORD **)v15);
  return result;
}
