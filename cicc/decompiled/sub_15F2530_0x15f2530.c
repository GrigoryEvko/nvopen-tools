// Function: sub_15F2530
// Address: 0x15f2530
//
unsigned __int64 __fastcall sub_15F2530(unsigned __int8 *a1, __int64 a2, char a3)
{
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  unsigned __int64 result; // rax
  int v7; // edx
  char v8; // dl
  int v9; // eax
  int v10; // esi
  unsigned __int64 v11; // rax
  int v12; // edx
  int v13; // eax
  char v14; // r12
  char v15; // al
  void *v16; // rdx

  if ( a3 )
  {
    v4 = a1[16];
    if ( (unsigned __int8)v4 <= 0x2Fu )
    {
      v5 = 0x80A800000000LL;
      if ( _bittest64(&v5, v4) )
      {
        result = *(unsigned __int8 *)(a2 + 16);
        if ( (unsigned __int8)result <= 0x17u )
        {
          if ( (_BYTE)result != 5 )
            return result;
          v11 = *(unsigned __int16 *)(a2 + 18);
          if ( (unsigned __int16)v11 > 0x17u || (v16 = &loc_80A800, !_bittest64((const __int64 *)&v16, v11)) )
          {
LABEL_28:
            if ( (unsigned __int16)(v11 - 24) > 1u && (unsigned int)(v11 - 17) > 1 )
              goto LABEL_32;
            v12 = a1[16];
            if ( (unsigned int)(v12 - 41) > 1 && (unsigned __int8)(v12 - 48) > 1u )
              goto LABEL_32;
            goto LABEL_13;
          }
        }
        else if ( (unsigned __int8)result > 0x2Fu || !_bittest64(&v5, result) )
        {
          goto LABEL_9;
        }
        sub_15F2330((__int64)a1, (*(_BYTE *)(a2 + 17) & 4) != 0);
        sub_15F2310((__int64)a1, (*(_BYTE *)(a2 + 17) & 2) != 0);
      }
    }
  }
  result = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)result <= 0x17u )
  {
    if ( (_BYTE)result != 5 )
      return result;
    LODWORD(v11) = *(unsigned __int16 *)(a2 + 18);
    goto LABEL_28;
  }
LABEL_9:
  if ( (unsigned int)(unsigned __int8)result - 41 > 1 && (unsigned __int8)(result - 48) > 1u
    || (v7 = a1[16], (unsigned __int8)(v7 - 48) > 1u) && (unsigned int)(v7 - 41) > 1 )
  {
LABEL_14:
    v8 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
    if ( v8 == 16 )
      v8 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a2 + 16LL) + 8LL);
    if ( (unsigned __int8)(v8 - 1) > 5u && (_BYTE)result != 76 )
      goto LABEL_24;
    goto LABEL_18;
  }
LABEL_13:
  sub_15F2350((__int64)a1, (*(_BYTE *)(a2 + 17) & 2) != 0);
  result = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)result > 0x17u )
    goto LABEL_14;
  if ( (_BYTE)result != 5 )
    return result;
LABEL_32:
  v13 = *(unsigned __int8 *)(*(_QWORD *)a2 + 8LL);
  if ( (_BYTE)v13 == 16 )
    v13 = *(unsigned __int8 *)(**(_QWORD **)(*(_QWORD *)a2 + 16LL) + 8LL);
  result = (unsigned int)(v13 - 1);
  if ( (unsigned __int8)result <= 5u || *(_WORD *)(a2 + 18) == 52 )
  {
LABEL_18:
    v9 = *(unsigned __int8 *)(*(_QWORD *)a1 + 8LL);
    if ( (_BYTE)v9 == 16 )
      v9 = *(unsigned __int8 *)(**(_QWORD **)(*(_QWORD *)a1 + 16LL) + 8LL);
    result = (unsigned int)(v9 - 1);
    if ( (unsigned __int8)result > 5u && a1[16] != 76 )
    {
      if ( *(_BYTE *)(a2 + 16) != 56 )
        return result;
      goto LABEL_39;
    }
    v10 = *(_BYTE *)(a2 + 17) >> 1;
    if ( v10 == 127 )
      v10 = -1;
    result = sub_15F2460((__int64)a1, v10);
LABEL_24:
    if ( *(_BYTE *)(a2 + 16) != 56 )
      return result;
LABEL_39:
    if ( a1[16] == 56 )
    {
      v14 = sub_15FA300(a2);
      v15 = sub_15FA300(a1);
      return sub_15FA2E0(a1, (unsigned __int8)(v15 | v14));
    }
  }
  return result;
}
