// Function: sub_111E110
// Address: 0x111e110
//
__int64 __fastcall sub_111E110(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  __int64 v5; // r9
  __int64 v6; // rsi
  __int64 v7; // rcx
  int v8; // r11d
  int v9; // r10d

  v2 = *(_QWORD *)(a2 + 16);
  v3 = 0;
  if ( !v2 || *(_QWORD *)(v2 + 8) || *(_BYTE *)a2 != 78 )
    return 0;
  v5 = *(_QWORD *)(a2 - 32);
  v6 = *(_QWORD *)(a2 + 8);
  v7 = *(_QWORD *)(v5 + 8);
  v8 = *(unsigned __int8 *)(v6 + 8);
  v9 = *(unsigned __int8 *)(v7 + 8);
  if ( (unsigned int)(v8 - 17) <= 1 == (unsigned int)(v9 - 17) <= 1 )
  {
    if ( (unsigned int)(v9 - 17) > 1
      || *(_DWORD *)(v6 + 32) == *(_DWORD *)(v7 + 32) && ((_BYTE)v8 == 18) == ((_BYTE)v9 == 18) )
    {
      v3 = 1;
      **a1 = v5;
    }
    else
    {
      return 0;
    }
  }
  return v3;
}
