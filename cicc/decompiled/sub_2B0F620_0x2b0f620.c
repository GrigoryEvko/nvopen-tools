// Function: sub_2B0F620
// Address: 0x2b0f620
//
unsigned __int64 __fastcall sub_2B0F620(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r15
  __int64 v5; // rax
  int v6; // [rsp+Ch] [rbp-34h]

  v2 = a2;
  v3 = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)(v3 + 8) <= 0x40u )
  {
    if ( !*(_QWORD *)v3 )
      return v2;
  }
  else
  {
    v6 = *(_DWORD *)(v3 + 8);
    if ( v6 == (unsigned int)sub_C444A0(*(_QWORD *)(a1 + 8)) )
      return v2;
  }
  v5 = sub_DFAAD0(*(__int64 **)a1, **(_QWORD **)(a1 + 16), v3, 0, 1u);
  v2 = a2 - v5;
  if ( __OFSUB__(a2, v5) )
  {
    v2 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v5 > 0 )
      return 0x8000000000000000LL;
  }
  return v2;
}
