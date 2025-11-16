// Function: sub_80B1D0
// Address: 0x80b1d0
//
_BOOL8 __fastcall sub_80B1D0(__int64 a1)
{
  __int64 **v1; // rbx
  __int64 *v3; // r12
  __int64 v4; // rcx
  __int64 *v5; // rsi
  __int64 v6; // r8
  __int64 *v7; // rbx
  __int64 v8; // rax

  for ( ; *(_BYTE *)(a1 + 140) == 12; a1 = *(_QWORD *)(a1 + 160) )
    ;
  v1 = *(__int64 ***)(*(_QWORD *)(a1 + 168) + 168LL);
  if ( v1
    && !*((_BYTE *)v1 + 8)
    && ((v3 = v1[4], v5 = sub_72BA30(0), v3 == v5) || (unsigned int)sub_8D97D0(v3, v5, 0, v4, v6))
    && (v7 = *v1) != 0
    && !*((_BYTE *)v7 + 8)
    && (unsigned int)sub_809870(v7[4], "char_traits")
    && (v8 = *v7) != 0
    && !*(_BYTE *)(v8 + 8)
    && !*(_QWORD *)v8 )
  {
    return (unsigned int)sub_809870(*(_QWORD *)(v8 + 32), "allocator") != 0;
  }
  else
  {
    return 0;
  }
}
