// Function: sub_2C763F0
// Address: 0x2c763f0
//
__int64 __fastcall sub_2C763F0(unsigned int a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 result; // rax
  unsigned __int64 v4; // rdx

  v2 = *(_QWORD *)(a2 + 32);
  result = a1;
  v4 = *(_QWORD *)(a2 + 24) - v2;
  if ( a1 == 1 )
  {
    if ( v4 <= 8 )
    {
      return sub_CB6200(a2, (unsigned __int8 *)"Warning: ", 9u);
    }
    else
    {
      *(_BYTE *)(v2 + 8) = 32;
      *(_QWORD *)v2 = 0x3A676E696E726157LL;
      *(_QWORD *)(a2 + 32) += 9LL;
      return 0x3A676E696E726157LL;
    }
  }
  else if ( a1 == 2 )
  {
    if ( v4 <= 5 )
    {
      return sub_CB6200(a2, "Info: ", 6u);
    }
    else
    {
      *(_DWORD *)v2 = 1868983881;
      *(_WORD *)(v2 + 4) = 8250;
      *(_QWORD *)(a2 + 32) += 6LL;
      return 8250;
    }
  }
  else if ( v4 <= 6 )
  {
    return sub_CB6200(a2, (unsigned __int8 *)"Error: ", 7u);
  }
  else
  {
    *(_DWORD *)v2 = 1869771333;
    *(_WORD *)(v2 + 4) = 14962;
    *(_BYTE *)(v2 + 6) = 32;
    *(_QWORD *)(a2 + 32) += 7LL;
  }
  return result;
}
