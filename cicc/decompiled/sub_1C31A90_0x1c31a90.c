// Function: sub_1C31A90
// Address: 0x1c31a90
//
__int64 __fastcall sub_1C31A90(unsigned int a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 result; // rax
  unsigned __int64 v4; // rdx

  v2 = *(_QWORD *)(a2 + 24);
  result = a1;
  v4 = *(_QWORD *)(a2 + 16) - v2;
  if ( a1 == 1 )
  {
    if ( v4 <= 8 )
    {
      return sub_16E7EE0(a2, "Warning: ", 9u);
    }
    else
    {
      *(_BYTE *)(v2 + 8) = 32;
      *(_QWORD *)v2 = 0x3A676E696E726157LL;
      *(_QWORD *)(a2 + 24) += 9LL;
      return 0x3A676E696E726157LL;
    }
  }
  else if ( a1 == 2 )
  {
    if ( v4 <= 5 )
    {
      return sub_16E7EE0(a2, "Info: ", 6u);
    }
    else
    {
      *(_DWORD *)v2 = 1868983881;
      *(_WORD *)(v2 + 4) = 8250;
      *(_QWORD *)(a2 + 24) += 6LL;
      return 8250;
    }
  }
  else if ( v4 <= 6 )
  {
    return sub_16E7EE0(a2, "Error: ", 7u);
  }
  else
  {
    *(_DWORD *)v2 = 1869771333;
    *(_WORD *)(v2 + 4) = 14962;
    *(_BYTE *)(v2 + 6) = 32;
    *(_QWORD *)(a2 + 24) += 7LL;
  }
  return result;
}
