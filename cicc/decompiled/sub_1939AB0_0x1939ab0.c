// Function: sub_1939AB0
// Address: 0x1939ab0
//
void __fastcall sub_1939AB0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 *v6; // rbx
  __int64 *v7; // r14
  __int64 v8; // rsi

  if ( *(_BYTE *)(a2 + 16) > 0x17u && !sub_15CCEE0(*a1, a2, a3) )
  {
    v5 = 3LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    {
      v6 = *(__int64 **)(a2 - 8);
      v7 = &v6[v5];
    }
    else
    {
      v7 = (__int64 *)a2;
      v6 = (__int64 *)(a2 - v5 * 8);
    }
    while ( v7 != v6 )
    {
      v8 = *v6;
      v6 += 3;
      sub_1939AB0(a1, v8, a3);
    }
    sub_15F22F0((_QWORD *)a2, a3);
  }
}
