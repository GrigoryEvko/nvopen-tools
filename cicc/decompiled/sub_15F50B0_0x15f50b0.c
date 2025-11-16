// Function: sub_15F50B0
// Address: 0x15f50b0
//
const char *__fastcall sub_15F50B0(__int64 *a1, __int64 a2, _QWORD *a3)
{
  const char *result; // rax
  __int64 v4; // r12
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // rax

  result = "both values to select must have same type";
  if ( *a3 == *(_QWORD *)a2 )
  {
    result = "select values cannot have token type";
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 10 )
    {
      v4 = *a1;
      if ( *(_BYTE *)(*a1 + 8) == 16 )
      {
        v5 = *(_QWORD *)(v4 + 24);
        v6 = sub_16498A0(a1);
        v7 = sub_1643320(v6);
        result = "vector select condition element type must be i1";
        if ( v5 == v7 )
        {
          result = "selected values for vector select must be vectors";
          if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
          {
            result = "vector select requires selected vectors to have the same vector length as select condition";
            if ( *(_QWORD *)(v4 + 32) == *(_QWORD *)(*(_QWORD *)a2 + 32LL) )
              return 0;
          }
        }
      }
      else
      {
        v8 = sub_16498A0(a1);
        if ( v4 == sub_1643320(v8) )
          return 0;
        return "select condition must be i1 or <n x i1>";
      }
    }
  }
  return result;
}
