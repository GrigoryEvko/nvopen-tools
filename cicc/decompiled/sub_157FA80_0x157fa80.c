// Function: sub_157FA80
// Address: 0x157fa80
//
__int64 __fastcall sub_157FA80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r14
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 result; // rax
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rax

  if ( a3 )
  {
    v5 = a3 + 24;
    sub_157F970(a1, a2);
    if ( (*(_BYTE *)(a1 + 23) & 0x20) != 0 )
    {
      v9 = *(_QWORD *)(a2 + 104);
      if ( v9 )
        sub_164D6D0(v9, a1);
    }
    v6 = *(_QWORD *)(a3 + 24);
    v7 = *(_QWORD *)(a1 + 24);
    *(_QWORD *)(a1 + 32) = v5;
    v6 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 24) = v6 | v7 & 7;
    *(_QWORD *)(v6 + 8) = a1 + 24;
    result = *(_QWORD *)(a3 + 24) & 7LL;
    *(_QWORD *)(a3 + 24) = result | (a1 + 24);
  }
  else
  {
    sub_157F970(a1, a2);
    if ( (*(_BYTE *)(a1 + 23) & 0x20) != 0 )
    {
      v10 = *(_QWORD *)(a2 + 104);
      if ( v10 )
        sub_164D6D0(v10, a1);
    }
    v11 = *(_QWORD *)(a2 + 72);
    v12 = *(_QWORD *)(a1 + 24);
    *(_QWORD *)(a1 + 32) = a2 + 72;
    v11 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 24) = v11 | v12 & 7;
    *(_QWORD *)(v11 + 8) = a1 + 24;
    result = *(_QWORD *)(a2 + 72) & 7LL;
    *(_QWORD *)(a2 + 72) = result | (a1 + 24);
  }
  return result;
}
