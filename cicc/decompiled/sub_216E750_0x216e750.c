// Function: sub_216E750
// Address: 0x216e750
//
char __fastcall sub_216E750(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // al
  __int64 v3; // rax
  unsigned __int8 v4; // cl
  int v5; // edx
  unsigned __int64 v6; // rdx
  __int64 v7; // rsi

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 == 17 )
  {
    LODWORD(v3) = sub_1C2F070(*(_QWORD *)(a2 + 24)) ^ 1;
    return v3;
  }
  if ( v2 <= 0x17u )
    goto LABEL_13;
  if ( v2 == 54 )
  {
    v3 = **(_QWORD **)(a2 - 24);
    if ( *(_BYTE *)(v3 + 8) == 16 )
      v3 = **(_QWORD **)(v3 + 16);
    LOBYTE(v3) = *(_DWORD *)(v3 + 8) >> 8 == 5 || *(_DWORD *)(v3 + 8) >> 8 == 0;
    return v3;
  }
  LOBYTE(v3) = sub_15F32D0(a2);
  if ( (_BYTE)v3 )
    return v3;
  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 <= 0x17u )
  {
LABEL_13:
    LOBYTE(v3) = 0;
    return v3;
  }
  if ( v4 != 78
    || (v3 = *(_QWORD *)(a2 - 24), *(_BYTE *)(v3 + 16))
    || (*(_BYTE *)(v3 + 33) & 0x20) == 0
    || (v5 = *(_DWORD *)(v3 + 36), LOBYTE(v3) = v5 == 4322 || (unsigned int)(v5 - 4344) <= 2, !(_BYTE)v3)
    && ((v6 = (unsigned int)(v5 - 3661), (unsigned int)v6 > 0x37)
     || (v7 = 0xC0000FF7F000FFLL, LOBYTE(v3) = 1, !_bittest64(&v7, v6))) )
  {
    LOBYTE(v3) = v4 == 78;
  }
  return v3;
}
