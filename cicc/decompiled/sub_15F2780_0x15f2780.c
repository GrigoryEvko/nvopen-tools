// Function: sub_15F2780
// Address: 0x15f2780
//
unsigned __int64 __fastcall sub_15F2780(unsigned __int8 *a1, __int64 a2)
{
  unsigned __int64 result; // rax
  __int64 v4; // rdx
  unsigned __int64 v5; // rcx
  int v6; // ecx
  bool v7; // al
  char v8; // dl
  int v9; // eax
  int v10; // esi
  int v11; // eax
  unsigned __int64 v12; // rdx
  int v13; // ecx
  void *v14; // rsi
  __int64 v15; // rdx
  bool v16; // al
  bool v17; // al
  int v18; // eax
  char v19; // r12
  char v20; // al

  result = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)result <= 0x17u )
  {
    if ( (_BYTE)result != 5 )
      return result;
    v12 = *(unsigned __int16 *)(a2 + 18);
    v13 = (unsigned __int16)v12;
    if ( (unsigned __int16)v12 > 0x17u || (v14 = &loc_80A800, !_bittest64((const __int64 *)&v14, v12)) )
    {
LABEL_33:
      if ( (unsigned int)(v13 - 17) > 1 && (unsigned __int16)(v12 - 24) > 1u )
        goto LABEL_35;
      goto LABEL_7;
    }
    v5 = a1[16];
    if ( (unsigned __int8)v5 > 0x2Fu )
    {
LABEL_32:
      v13 = (unsigned __int16)v12;
      goto LABEL_33;
    }
  }
  else
  {
    if ( (unsigned __int8)result > 0x2Fu )
      goto LABEL_5;
    v4 = 0x80A800000000LL;
    if ( !_bittest64(&v4, result) )
      goto LABEL_5;
    v5 = a1[16];
    if ( (unsigned __int8)v5 > 0x2Fu )
      goto LABEL_5;
  }
  v15 = 0x80A800000000LL;
  if ( _bittest64(&v15, v5) )
  {
    v16 = sub_15F2380((__int64)a1);
    sub_15F2330((__int64)a1, v16 & (*(_BYTE *)(a2 + 17) >> 2) & 1);
    v17 = sub_15F2370((__int64)a1);
    sub_15F2310((__int64)a1, v17 & (*(_BYTE *)(a2 + 17) >> 1) & 1);
    result = *(unsigned __int8 *)(a2 + 16);
  }
  if ( (unsigned __int8)result <= 0x17u )
  {
    if ( (_BYTE)result != 5 )
      return result;
    LOWORD(v12) = *(_WORD *)(a2 + 18);
    goto LABEL_32;
  }
LABEL_5:
  if ( (unsigned int)(unsigned __int8)result - 41 > 1 && (unsigned __int8)(result - 48) > 1u )
    goto LABEL_11;
LABEL_7:
  v6 = a1[16];
  if ( (unsigned int)(v6 - 41) <= 1 || (unsigned __int8)(v6 - 48) <= 1u )
  {
    v7 = sub_15F23D0((__int64)a1);
    sub_15F2350((__int64)a1, v7 & (*(_BYTE *)(a2 + 17) >> 1) & 1);
    result = *(unsigned __int8 *)(a2 + 16);
  }
  if ( (unsigned __int8)result > 0x17u )
  {
LABEL_11:
    v8 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
    if ( v8 == 16 )
      v8 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a2 + 16LL) + 8LL);
    if ( (unsigned __int8)(v8 - 1) > 5u && (_BYTE)result != 76 )
      goto LABEL_21;
    goto LABEL_15;
  }
  if ( (_BYTE)result != 5 )
    return result;
LABEL_35:
  v18 = *(unsigned __int8 *)(*(_QWORD *)a2 + 8LL);
  if ( (_BYTE)v18 == 16 )
    v18 = *(unsigned __int8 *)(**(_QWORD **)(*(_QWORD *)a2 + 16LL) + 8LL);
  result = (unsigned int)(v18 - 1);
  if ( (unsigned __int8)result <= 5u || *(_WORD *)(a2 + 18) == 52 )
  {
LABEL_15:
    v9 = *(unsigned __int8 *)(*(_QWORD *)a1 + 8LL);
    if ( (_BYTE)v9 == 16 )
      v9 = *(unsigned __int8 *)(**(_QWORD **)(*(_QWORD *)a1 + 16LL) + 8LL);
    result = (unsigned int)(v9 - 1);
    if ( (unsigned __int8)result > 5u && a1[16] != 76 )
    {
      if ( *(_BYTE *)(a2 + 16) != 56 )
        return result;
      goto LABEL_42;
    }
    v10 = sub_15F24E0((__int64)a1);
    v11 = *(_BYTE *)(a2 + 17) >> 1;
    if ( v11 != 127 )
      v10 &= v11;
    result = sub_15F2460((__int64)a1, v10);
LABEL_21:
    if ( *(_BYTE *)(a2 + 16) != 56 )
      return result;
LABEL_42:
    if ( a1[16] == 56 )
    {
      v19 = sub_15FA300(a2);
      v20 = sub_15FA300(a1);
      return sub_15FA2E0(a1, (unsigned __int8)(v20 & v19));
    }
  }
  return result;
}
