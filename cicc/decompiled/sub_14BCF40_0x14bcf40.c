// Function: sub_14BCF40
// Address: 0x14bcf40
//
bool *__fastcall sub_14BCF40(bool *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5, int a6)
{
  bool *result; // rax
  unsigned __int8 v8; // cl
  bool *v9; // [rsp-10h] [rbp-10h]

  result = a1;
  if ( a2 == a3 )
  {
    a1[1] = 1;
    *a1 = a5;
  }
  else
  {
    v8 = *(_BYTE *)(a3 + 16);
    if ( v8 > 0x17u && (unsigned __int8)(v8 - 75) <= 1u )
    {
      v9 = result;
      sub_14BC8D0(result, a2, *(_WORD *)(a3 + 18) & 0x7FFF, *(_QWORD *)(a3 - 48), *(_QWORD *)(a3 - 24), a4, a5, a6);
      return v9;
    }
    else
    {
      result[1] = 0;
    }
  }
  return result;
}
