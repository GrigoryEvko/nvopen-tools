// Function: sub_33549A0
// Address: 0x33549a0
//
void __fastcall sub_33549A0(__int64 a1, __int64 a2)
{
  int v3; // eax
  _BYTE *v4; // rsi
  __int64 v5; // [rsp+8h] [rbp-8h] BYREF

  v3 = *(_DWORD *)(a1 + 40);
  v5 = a2;
  *(_DWORD *)(a1 + 40) = ++v3;
  *(_DWORD *)(a2 + 204) = v3;
  v4 = *(_BYTE **)(a1 + 24);
  if ( v4 == *(_BYTE **)(a1 + 32) )
  {
    sub_2ECAD30(a1 + 16, v4, &v5);
  }
  else
  {
    if ( v4 )
    {
      *(_QWORD *)v4 = a2;
      v4 = *(_BYTE **)(a1 + 24);
    }
    *(_QWORD *)(a1 + 24) = v4 + 8;
  }
}
