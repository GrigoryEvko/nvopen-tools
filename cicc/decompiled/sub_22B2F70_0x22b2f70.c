// Function: sub_22B2F70
// Address: 0x22b2f70
//
__int64 __fastcall sub_22B2F70(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int16 v3; // dx
  __int64 result; // rax
  unsigned int v5; // ecx

  v2 = *(_QWORD *)(a2 - 32);
  if ( !v2 || *(_BYTE *)v2 || *(_QWORD *)(v2 + 24) != *(_QWORD *)(a2 + 80) )
  {
    if ( !sub_B491E0(a2) )
      return 1;
    goto LABEL_5;
  }
  v5 = *(_DWORD *)(v2 + 36);
  if ( v5 >= 0x48 )
    return sub_22AE6A0(a1, a2);
  if ( v5 > 0x44 )
    return 2;
  if ( v5 )
    return sub_22AE6A0(a1, a2);
  if ( sub_B491E0(a2) )
  {
LABEL_5:
    if ( *(_BYTE *)(a1 + 1) )
      goto LABEL_6;
    return 1;
  }
LABEL_6:
  v3 = *(_WORD *)(a2 + 2);
  if ( ((((v3 >> 2) & 0x3FF) - 18) & 0xFFFD) == 0 && !*(_BYTE *)(a1 + 3) )
    return 1;
  result = 0;
  if ( (v3 & 3) == 2 )
    return *(_BYTE *)(a1 + 3) ^ 1u;
  return result;
}
