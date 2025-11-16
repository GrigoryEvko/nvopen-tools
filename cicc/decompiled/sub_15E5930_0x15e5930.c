// Function: sub_15E5930
// Address: 0x15e5930
//
void __fastcall sub_15E5930(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdi

  if ( *(_QWORD *)(a1 - 24) )
  {
    v2 = *(_QWORD *)(a1 - 16);
    v3 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v3 = v2;
    if ( v2 )
      *(_QWORD *)(v2 + 16) = *(_QWORD *)(v2 + 16) & 3LL | v3;
  }
  *(_QWORD *)(a1 - 24) = a2;
  if ( a2 )
  {
    v4 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(a1 - 16) = v4;
    if ( v4 )
      *(_QWORD *)(v4 + 16) = (a1 - 16) | *(_QWORD *)(v4 + 16) & 3LL;
    v5 = *(_QWORD *)(a1 - 8);
    v6 = a1 - 24;
    *(_QWORD *)(v6 + 16) = (a2 + 8) | v5 & 3;
    *(_QWORD *)(a2 + 8) = v6;
  }
}
