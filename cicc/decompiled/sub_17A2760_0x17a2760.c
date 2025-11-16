// Function: sub_17A2760
// Address: 0x17a2760
//
void __fastcall sub_17A2760(__int64 a1, unsigned int a2)
{
  unsigned int v2; // eax

  v2 = *(_DWORD *)(a1 + 8);
  if ( v2 > 0x40 )
  {
    sub_16A8110(a1, a2);
  }
  else if ( v2 == a2 )
  {
    *(_QWORD *)a1 = 0;
  }
  else
  {
    *(_QWORD *)a1 >>= a2;
  }
}
