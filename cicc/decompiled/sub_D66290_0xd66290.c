// Function: sub_D66290
// Address: 0xd66290
//
__int64 __fastcall sub_D66290(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  void *v3; // rdx
  char *v4; // rcx
  __int64 result; // rax
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // r13
  __m128i si128; // xmm0
  size_t v9; // rdx
  char *v10; // rsi
  __int64 v11; // rdx

  v2 = a2;
  v3 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v3 <= 0xDu )
  {
    sub_CB6200(a2, "LocationSize::", 0xEu);
    v4 = *(char **)(a2 + 32);
  }
  else
  {
    qmemcpy(v3, "LocationSize::", 14);
    v4 = (char *)(*(_QWORD *)(a2 + 32) + 14LL);
    *(_QWORD *)(a2 + 32) = v4;
  }
  result = *a1;
  v6 = *(_QWORD *)(a2 + 24) - (_QWORD)v4;
  if ( *a1 == -1 )
  {
    if ( v6 > 0x13 )
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F725C0);
      *((_DWORD *)v4 + 4) = 1919251566;
      *(__m128i *)v4 = si128;
      *(_QWORD *)(a2 + 32) += 20LL;
      return result;
    }
    v9 = 20;
    v10 = "beforeOrAfterPointer";
    return sub_CB6200(v2, (unsigned __int8 *)v10, v9);
  }
  switch ( result )
  {
    case -4611686018427387906LL:
      if ( v6 > 0xB )
      {
        qmemcpy(v4, "afterPointer", 12);
        *(_QWORD *)(a2 + 32) += 12LL;
        return 0x696F507265746661LL;
      }
      v9 = 12;
      v10 = "afterPointer";
      return sub_CB6200(v2, (unsigned __int8 *)v10, v9);
    case -3LL:
      if ( v6 > 7 )
      {
        *(_QWORD *)v4 = 0x7974706D4570616DLL;
        *(_QWORD *)(a2 + 32) += 8LL;
        return 0x7974706D4570616DLL;
      }
      v9 = 8;
      v10 = "mapEmpty";
      return sub_CB6200(v2, (unsigned __int8 *)v10, v9);
    case -4LL:
      if ( v6 > 0xB )
      {
        qmemcpy(v4, "mapTombstone", 12);
        *(_QWORD *)(a2 + 32) += 12LL;
        return 0x73626D6F5470616DLL;
      }
      v9 = 12;
      v10 = "mapTombstone";
      return sub_CB6200(v2, (unsigned __int8 *)v10, v9);
  }
  if ( result < 0 )
  {
    if ( v6 <= 0xA )
    {
      v2 = sub_CB6200(a2, "upperBound(", 0xBu);
    }
    else
    {
      qmemcpy(v4, "upperBound(", 11);
      *(_QWORD *)(a2 + 32) += 11LL;
    }
  }
  else if ( v6 <= 7 )
  {
    v2 = sub_CB6200(a2, "precise(", 8u);
  }
  else
  {
    *(_QWORD *)v4 = 0x2865736963657270LL;
    *(_QWORD *)(a2 + 32) += 8LL;
  }
  v7 = *a1 & 0x3FFFFFFFFFFFFFFFLL;
  if ( (*a1 & 0x4000000000000000LL) != 0 )
  {
    v11 = *(_QWORD *)(v2 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v2 + 24) - v11) <= 8 )
    {
      sub_CB6200(v2, "vscale x ", 9u);
    }
    else
    {
      *(_BYTE *)(v11 + 8) = 32;
      *(_QWORD *)v11 = 0x7820656C61637376LL;
      *(_QWORD *)(v2 + 32) += 9LL;
    }
  }
  sub_CB59D0(v2, v7);
  result = *(_QWORD *)(v2 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(v2 + 24) )
    return sub_CB5D20(v2, 41);
  *(_QWORD *)(v2 + 32) = result + 1;
  *(_BYTE *)result = 41;
  return result;
}
