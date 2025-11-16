// Function: sub_5CFA80
// Address: 0x5cfa80
//
__int64 __fastcall sub_5CFA80(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  char v5; // al
  char *v6; // rax
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // rbx
  char *v10; // rax

  if ( a3 != 6 )
  {
    if ( a3 == 3 && *(_BYTE *)(a1 + 9) == 3 )
      goto LABEL_6;
    goto LABEL_3;
  }
  v5 = *(_BYTE *)(a2 + 140);
  if ( (unsigned __int8)(v5 - 9) <= 2u )
  {
    *(_BYTE *)(*(_QWORD *)(a2 + 168) + 112LL) &= ~4u;
    v5 = *(_BYTE *)(a2 + 140);
    if ( (unsigned __int8)(v5 - 9) <= 2u )
    {
LABEL_9:
      if ( (*(_DWORD *)(a1 + 8) & 0xFFFF00) == 0x20300 && (*(_BYTE *)(a2 + 177) & 4) != 0 )
      {
        v6 = sub_5C79F0(a1);
        sub_684B10(1874, a1 + 56, v6);
        *(_BYTE *)(a1 + 8) = 0;
        return a2;
      }
      goto LABEL_3;
    }
  }
  if ( v5 != 2 )
  {
    if ( v5 == 12 && *(_QWORD *)(a2 + 8) )
      goto LABEL_9;
LABEL_20:
    sub_5CCAE0(5u, a1);
    goto LABEL_3;
  }
  if ( (*(_BYTE *)(a2 + 161) & 8) == 0 )
    goto LABEL_20;
  if ( (*(_DWORD *)(a1 + 8) & 0xFFFF00) == 0x20300 )
  {
    sub_684B30(1723, a1 + 56);
LABEL_6:
    *(_BYTE *)(a1 + 8) = 0;
    return a2;
  }
LABEL_3:
  if ( !*(_BYTE *)(a1 + 8) )
    return a2;
  v7 = sub_72A270(a2, a3);
  v8 = *(_QWORD *)(a1 + 32);
  v9 = v7;
  if ( !v8 )
    goto LABEL_25;
  if ( *(_BYTE *)(a1 + 9) == 2 && dword_4F077B8 && qword_4F077A8 <= 0x9E33u && !dword_4F077B4 )
  {
    v10 = sub_5C79F0(a1);
    sub_6851A0(1099, v8 + 24, v10);
    *(_BYTE *)(a1 + 8) = 0;
LABEL_25:
    if ( !v9 )
      return a2;
    goto LABEL_16;
  }
  if ( !v7 )
    return a2;
  sub_5CF8B0(*(_BYTE *)(a1 + 8), v7, *(_QWORD *)(v8 + 40), v8 + 24);
LABEL_16:
  if ( !*(_BYTE *)(a1 + 8) )
    return a2;
  *(_BYTE *)(v9 + 90) |= 0x80u;
  return a2;
}
