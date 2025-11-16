// Function: sub_989280
// Address: 0x989280
//
__int64 __fastcall sub_989280(int *a1, int *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  char v7; // dl
  __int64 v8; // rdi
  __int64 v9; // rax
  int v10; // edx
  char v11; // cl
  int v12; // edx

  result = (unsigned int)*a2;
  v7 = *a2;
  *a1 = result;
  if ( (v7 & 0x60) != 0x60 && (result & 0x90) != 0 )
  {
    v8 = a4;
    if ( (unsigned int)*(unsigned __int8 *)(a4 + 8) - 17 <= 1 )
      v8 = **(_QWORD **)(a4 + 16);
    v9 = sub_BCAC60(v8);
    result = sub_B2DB90(a3, v9);
    v10 = *a2;
    v11 = BYTE1(result);
    if ( (*a2 & 0x80u) != 0 )
    {
      if ( !(_WORD)result )
        return result;
      *a1 |= 0x40u;
      v10 = *a2;
    }
    if ( (v10 & 0x10) != 0 )
    {
      if ( (_BYTE)result )
      {
        if ( (_BYTE)result == 2 )
        {
          v12 = *a1;
          result = *a1 | 0x20u;
          if ( v11 != 2 )
            v12 = *a1 | 0x20;
          goto LABEL_13;
        }
      }
      else if ( !BYTE1(result) )
      {
        return result;
      }
      result = (unsigned int)(result - 2);
      v12 = *a1 | 0x20;
      *a1 = v12;
      if ( (unsigned __int8)result <= 1u || (unsigned __int8)(v11 - 2) <= 1u )
LABEL_13:
        *a1 = v12 | 0x40;
    }
  }
  return result;
}
