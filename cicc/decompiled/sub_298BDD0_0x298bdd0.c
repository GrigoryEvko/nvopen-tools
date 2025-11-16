// Function: sub_298BDD0
// Address: 0x298bdd0
//
void __fastcall sub_298BDD0(__int64 a1, __int64 a2)
{
  _BYTE *v3; // rsi
  __int64 *v4; // r13
  __int64 *i; // rbx
  __int64 v6; // rdi
  _QWORD v7[5]; // [rsp+8h] [rbp-28h] BYREF

  v7[0] = a1;
  v3 = *(_BYTE **)(a2 + 8);
  if ( v3 == *(_BYTE **)(a2 + 16) )
  {
    sub_22DCD60(a2, v3, v7);
  }
  else
  {
    if ( v3 )
    {
      *(_QWORD *)v3 = a1;
      v3 = *(_BYTE **)(a2 + 8);
    }
    *(_QWORD *)(a2 + 8) = v3 + 8;
  }
  v4 = *(__int64 **)(a1 + 48);
  for ( i = *(__int64 **)(a1 + 40); v4 != i; ++i )
  {
    v6 = *i;
    sub_298BDD0(v6, a2);
  }
}
