// Function: sub_15FB2A0
// Address: 0x15fb2a0
//
__int64 __fastcall sub_15FB2A0(__int64 a1, unsigned int *a2, __int64 a3)
{
  unsigned int *v3; // r12
  unsigned int *v4; // rbx
  char v5; // al
  __int64 v6; // rsi

  v3 = &a2[a3];
  if ( v3 == a2 )
    return a1;
  v4 = a2;
  while ( 1 )
  {
    v5 = *(_BYTE *)(a1 + 8);
    v6 = *v4;
    if ( v5 != 14 )
      break;
    if ( (unsigned __int64)(unsigned int)v6 >= *(_QWORD *)(a1 + 32) )
      return 0;
LABEL_5:
    ++v4;
    a1 = sub_1643D80(a1, v6);
    if ( v3 == v4 )
      return a1;
  }
  if ( v5 == 13 && (unsigned int)v6 < *(_DWORD *)(a1 + 12) )
    goto LABEL_5;
  return 0;
}
