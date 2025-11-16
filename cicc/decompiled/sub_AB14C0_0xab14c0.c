// Function: sub_AB14C0
// Address: 0xab14c0
//
__int64 __fastcall sub_AB14C0(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r13d
  __int64 v4; // rbx
  unsigned int v6; // eax

  if ( sub_AAF760(a2) || sub_AB0120(a2) )
  {
    v2 = *(_DWORD *)(a2 + 8);
    v3 = v2 - 1;
    *(_DWORD *)(a1 + 8) = v2;
    v4 = 1LL << ((unsigned __int8)v2 - 1);
    if ( v2 <= 0x40 )
    {
      *(_QWORD *)a1 = 0;
LABEL_5:
      *(_QWORD *)a1 |= v4;
      return a1;
    }
    sub_C43690(a1, 0, 0);
    if ( *(_DWORD *)(a1 + 8) <= 0x40u )
      goto LABEL_5;
    *(_QWORD *)(*(_QWORD *)a1 + 8LL * (v3 >> 6)) |= v4;
    return a1;
  }
  else
  {
    v6 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 8) = v6;
    if ( v6 > 0x40 )
    {
      sub_C43780(a1, a2);
      return a1;
    }
    *(_QWORD *)a1 = *(_QWORD *)a2;
    return a1;
  }
}
