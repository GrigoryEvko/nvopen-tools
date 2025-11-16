// Function: sub_1EBB580
// Address: 0x1ebb580
//
__int64 __fastcall sub_1EBB580(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  unsigned int v5; // r8d
  unsigned __int64 v6; // r11
  __int64 result; // rax
  unsigned int v8; // r10d
  unsigned int *v9; // rdx
  unsigned int v10; // ecx
  unsigned int *v11; // rsi

  v5 = a4;
  v6 = HIDWORD(a4);
  result = (a2 - 1) / 2;
  if ( a2 <= a3 )
  {
    *(_QWORD *)(a1 + 8 * a2) = a4;
  }
  else
  {
    v8 = a4;
    while ( 1 )
    {
      v9 = (unsigned int *)(a1 + 8 * result);
      v10 = *v9;
      if ( v8 <= *v9 && (v8 != v10 || (unsigned int)v6 <= v9[1]) )
        break;
      v11 = (unsigned int *)(a1 + 8 * a2);
      *v11 = v10;
      v11[1] = v9[1];
      a2 = result;
      if ( a3 >= result )
        goto LABEL_9;
      result = (result - 1) / 2;
    }
    v9 = (unsigned int *)(a1 + 8 * a2);
LABEL_9:
    *v9 = v5;
    v9[1] = v6;
  }
  return result;
}
