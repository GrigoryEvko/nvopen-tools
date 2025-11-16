// Function: sub_14A9290
// Address: 0x14a9290
//
__int64 __fastcall sub_14A9290(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned int v5; // eax

  v5 = *(_DWORD *)(a3 + 8);
  *(_DWORD *)(a1 + 8) = v5;
  if ( v5 > 0x40 )
  {
    sub_16A4FD0(a1, a3);
    v5 = *(_DWORD *)(a1 + 8);
    if ( v5 > 0x40 )
    {
      sub_16A8110(a1, a4);
      return a1;
    }
  }
  else
  {
    *(_QWORD *)a1 = *(_QWORD *)a3;
  }
  if ( a4 == v5 )
  {
    *(_QWORD *)a1 = 0;
    return a1;
  }
  else
  {
    *(_QWORD *)a1 >>= a4;
    return a1;
  }
}
