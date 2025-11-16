// Function: sub_7F5D80
// Address: 0x7f5d80
//
__int64 __fastcall sub_7F5D80(__int64 a1, int a2, __int64 a3)
{
  char i; // al
  __int64 result; // rax
  __int64 k; // rax
  __int64 j; // rax

  for ( i = *(_BYTE *)(a1 + 140); i == 12; i = *(_BYTE *)(a1 + 140) )
    a1 = *(_QWORD *)(a1 + 160);
  if ( i == 8 || i == 15 )
  {
    *(_DWORD *)a3 = 1;
    *(_QWORD *)(a3 + 8) = 0;
    *(_QWORD *)(a3 + 16) = 0;
    *(_QWORD *)(a3 + 24) = 0;
    *(_QWORD *)(a3 + 32) = 0;
    *(_DWORD *)(a3 + 40) = a2;
    if ( *(_BYTE *)(a1 + 140) == 8 )
    {
      for ( j = sub_8D4050(a1); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
        ;
      *(_QWORD *)(a3 + 24) = j;
      result = *(_QWORD *)(a1 + 176);
      *(_QWORD *)(a3 + 32) = result;
    }
    else
    {
      for ( k = *(_QWORD *)(a1 + 160); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
        ;
      *(_QWORD *)(a3 + 24) = k;
      result = sub_8D4620(a1);
      *(_QWORD *)(a3 + 32) = result;
    }
  }
  else
  {
    *(_DWORD *)a3 = 0;
    *(_QWORD *)(a3 + 8) = 0;
    *(_QWORD *)(a3 + 16) = 0;
    *(_QWORD *)(a3 + 24) = 0;
    *(_QWORD *)(a3 + 32) = 0;
    *(_DWORD *)(a3 + 40) = a2;
    result = sub_72FD90(*(_QWORD *)(a1 + 160), a2);
    if ( result )
    {
      *(_QWORD *)(a3 + 8) = result;
      result = *(_QWORD *)(result + 120);
      *(_QWORD *)(a3 + 24) = result;
    }
  }
  return result;
}
