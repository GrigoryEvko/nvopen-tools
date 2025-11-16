// Function: sub_1DD4D70
// Address: 0x1dd4d70
//
__int64 __fastcall sub_1DD4D70(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  __int16 v5; // r11
  unsigned __int16 v6; // r10
  __int64 v7; // r14
  unsigned __int64 v9; // rbx
  __int64 v10; // r12
  __int64 i; // rcx
  __int64 v12; // rdx
  __int64 result; // rax
  __int64 v14; // rcx
  unsigned __int16 *v15; // rsi

  v5 = a4;
  v6 = a4;
  v7 = a3 & 1;
  v9 = HIDWORD(a4);
  v10 = (a3 - 1) / 2;
  if ( a2 >= v10 )
  {
    result = a1 + 8 * a2;
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    v12 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v12 )
  {
    v12 = 2 * (i + 1);
    result = a1 + 16 * (i + 1);
    if ( *(_WORD *)result < *(_WORD *)(a1 + 8 * (v12 - 1)) )
      result = a1 + 8 * --v12;
    *(_QWORD *)(a1 + 8 * i) = *(_QWORD *)result;
    if ( v12 >= v10 )
      break;
  }
  if ( !v7 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v12 )
    {
      v12 = 2 * v12 + 1;
      *(_QWORD *)result = *(_QWORD *)(a1 + 8 * v12);
      result = a1 + 8 * v12;
    }
  }
  v14 = (v12 - 1) / 2;
  if ( v12 > a2 )
  {
    while ( 1 )
    {
      v15 = (unsigned __int16 *)(a1 + 8 * v14);
      result = a1 + 8 * v12;
      if ( *v15 >= v6 )
        break;
      *(_QWORD *)result = *(_QWORD *)v15;
      v12 = v14;
      if ( a2 >= v14 )
      {
        result = a1 + 8 * v14;
        break;
      }
      v14 = (v14 - 1) / 2;
    }
  }
LABEL_13:
  *(_WORD *)result = v5;
  *(_DWORD *)(result + 4) = v9;
  return result;
}
