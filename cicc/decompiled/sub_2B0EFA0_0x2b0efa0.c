// Function: sub_2B0EFA0
// Address: 0x2b0efa0
//
bool __fastcall sub_2B0EFA0(__int64 a1)
{
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // rdx
  _BYTE *v8; // rdi

  if ( *(_BYTE *)a1 != 86 )
    return 0;
  v3 = *(_QWORD *)(a1 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
    v3 = **(_QWORD **)(v3 + 16);
  if ( !sub_BCAC40(v3, 1) )
  {
LABEL_12:
    v4 = *(_QWORD *)(a1 + 8);
    goto LABEL_13;
  }
  if ( *(_BYTE *)a1 == 57 )
    return 1;
  v4 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)a1 == 86 && v4 == *(_QWORD *)(*(_QWORD *)(a1 - 96) + 8LL) && **(_BYTE **)(a1 - 32) <= 0x15u )
  {
    if ( sub_AC30F0(*(_QWORD *)(a1 - 32)) )
      return 1;
    goto LABEL_12;
  }
LABEL_13:
  if ( (unsigned int)*(unsigned __int8 *)(v4 + 8) - 17 <= 1 )
    v4 = **(_QWORD **)(v4 + 16);
  if ( !sub_BCAC40(v4, 1) )
    return 0;
  if ( *(_BYTE *)a1 == 58 )
    return 1;
  if ( *(_BYTE *)a1 == 86
    && (v7 = *(_QWORD *)(a1 + 8), *(_QWORD *)(*(_QWORD *)(a1 - 96) + 8LL) == v7)
    && (v8 = *(_BYTE **)(a1 - 64), *v8 <= 0x15u) )
  {
    return sub_AD7A80(v8, 1, v7, v5, v6);
  }
  else
  {
    return 0;
  }
}
