// Function: sub_AB13A0
// Address: 0xab13a0
//
__int64 __fastcall sub_AB13A0(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // ebx
  __int64 v4; // r13
  unsigned __int64 v5; // rdx
  __int64 v7; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-28h]

  if ( sub_AAF760(a2) || sub_AB01B0(a2) )
  {
    v2 = *(_DWORD *)(a2 + 8);
    v3 = v2 - 1;
    *(_DWORD *)(a1 + 8) = v2;
    v4 = ~(1LL << ((unsigned __int8)v2 - 1));
    if ( v2 <= 0x40 )
    {
      v5 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v2;
      if ( !v2 )
        v5 = 0;
      *(_QWORD *)a1 = v5;
      goto LABEL_7;
    }
    sub_C43690(a1, -1, 1);
    if ( *(_DWORD *)(a1 + 8) <= 0x40u )
    {
LABEL_7:
      *(_QWORD *)a1 &= v4;
      return a1;
    }
    *(_QWORD *)(*(_QWORD *)a1 + 8LL * (v3 >> 6)) &= v4;
    return a1;
  }
  else
  {
    v8 = *(_DWORD *)(a2 + 24);
    if ( v8 > 0x40 )
      sub_C43780(&v7, a2 + 16);
    else
      v7 = *(_QWORD *)(a2 + 16);
    sub_C46F20(&v7, 1);
    *(_DWORD *)(a1 + 8) = v8;
    *(_QWORD *)a1 = v7;
    return a1;
  }
}
