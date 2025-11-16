// Function: sub_E8D690
// Address: 0xe8d690
//
__int64 __fastcall sub_E8D690(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  _BYTE *v4; // rax

  v3 = *(_QWORD *)(a1 + 296);
  if ( v3 )
  {
    sub_E5BA60(v3, a2);
    v4 = *(_BYTE **)(*(_QWORD *)(a1 + 8) + 2368LL);
    if ( v4 )
      *(_BYTE *)(*(_QWORD *)(a1 + 296) + 33LL) = *v4 & 1;
  }
  *(_WORD *)(a1 + 304) = 1;
  return sub_E98300(a1);
}
