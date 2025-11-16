// Function: sub_AB0910
// Address: 0xab0910
//
__int64 __fastcall sub_AB0910(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned __int64 v3; // rdx
  __int64 v5; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v6; // [rsp+8h] [rbp-28h]

  if ( sub_AAF760(a2) || sub_AB0100(a2) )
  {
    v2 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 8) = v2;
    if ( v2 > 0x40 )
    {
      sub_C43690(a1, -1, 1);
      return a1;
    }
    else
    {
      v3 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v2;
      if ( !v2 )
        v3 = 0;
      *(_QWORD *)a1 = v3;
      return a1;
    }
  }
  else
  {
    v6 = *(_DWORD *)(a2 + 24);
    if ( v6 > 0x40 )
      sub_C43780(&v5, a2 + 16);
    else
      v5 = *(_QWORD *)(a2 + 16);
    sub_C46F20(&v5, 1);
    *(_DWORD *)(a1 + 8) = v6;
    *(_QWORD *)a1 = v5;
    return a1;
  }
}
