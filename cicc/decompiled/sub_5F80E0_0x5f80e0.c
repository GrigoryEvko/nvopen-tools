// Function: sub_5F80E0
// Address: 0x5f80e0
//
__int64 __fastcall sub_5F80E0(__int64 a1)
{
  __int64 result; // rax
  _QWORD *v3; // r12
  __int64 v4; // rsi
  __int64 i; // rdi
  __int64 v6; // rdi

  result = *(_QWORD *)(a1 + 152);
  v3 = *(_QWORD **)(result + 168);
  if ( *(_BYTE *)(a1 + 174) == 1 && !*v3 )
  {
    v6 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
    if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v6 + 96LL) + 183LL) & 0x10) != 0 )
    {
      result = sub_6851C0(2438, dword_4F07508);
      v3[7] = 0;
      return result;
    }
    result = sub_5E87D0(v6);
  }
  if ( v3[7] )
  {
    v3[7] = 0;
    v4 = 0;
    if ( (*(_BYTE *)(a1 + 194) & 0x40) != 0 )
      v4 = **(_QWORD **)(a1 + 232);
    result = sub_5F8DB0(a1, v4);
    if ( v3[7] )
    {
      for ( i = *(_QWORD *)(a1 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      result = sub_8D76D0();
      if ( (_DWORD)result )
        *(_BYTE *)(a1 + 195) |= 0x10u;
    }
  }
  return result;
}
