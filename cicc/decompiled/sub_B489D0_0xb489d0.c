// Function: sub_B489D0
// Address: 0xb489d0
//
const char *__fastcall sub_B489D0(__int64 a1, __int64 a2, __int64 a3)
{
  const char *result; // rax
  __int64 v5; // rdx
  __int64 v6; // r12
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // rsi
  int v11; // ecx
  __int64 v12; // rax

  result = "both values to select must have same type";
  v5 = *(_QWORD *)(a2 + 8);
  if ( *(_QWORD *)(a3 + 8) == v5 )
  {
    result = "select values cannot have token type";
    if ( *(_BYTE *)(v5 + 8) != 11 )
    {
      v6 = *(_QWORD *)(a1 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17 > 1 )
      {
        v12 = sub_BD5C60(a1, a2);
        if ( v6 != sub_BCB2A0(v12) )
          return "select condition must be i1 or <n x i1>";
        return 0;
      }
      v7 = *(_QWORD *)(v6 + 24);
      v8 = sub_BD5C60(a1, a2);
      v9 = sub_BCB2A0(v8);
      result = "vector select condition element type must be i1";
      if ( v7 == v9 )
      {
        v10 = *(_QWORD *)(a2 + 8);
        result = "selected values for vector select must be vectors";
        v11 = *(unsigned __int8 *)(v10 + 8);
        if ( (unsigned int)(v11 - 17) <= 1 )
        {
          if ( (*(_BYTE *)(v6 + 8) == 18) != ((_BYTE)v11 == 18) || *(_DWORD *)(v10 + 32) != *(_DWORD *)(v6 + 32) )
            return "vector select requires selected vectors to have the same vector length as select condition";
          return 0;
        }
      }
    }
  }
  return result;
}
