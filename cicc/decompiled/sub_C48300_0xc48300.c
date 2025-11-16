// Function: sub_C48300
// Address: 0xc48300
//
__int64 __fastcall sub_C48300(__int64 a1, __int64 a2, int a3)
{
  unsigned int v3; // eax
  unsigned int v4; // r13d

  v3 = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a1 + 8) = v3;
  v4 = v3 - a3;
  if ( v3 > 0x40 )
  {
    sub_C43780(a1, (const void **)a2);
    v3 = *(_DWORD *)(a1 + 8);
    if ( v3 > 0x40 )
    {
      sub_C482E0(a1, v4);
      return a1;
    }
  }
  else
  {
    *(_QWORD *)a1 = *(_QWORD *)a2;
  }
  if ( v4 == v3 )
  {
    *(_QWORD *)a1 = 0;
    return a1;
  }
  else
  {
    *(_QWORD *)a1 >>= v4;
    return a1;
  }
}
