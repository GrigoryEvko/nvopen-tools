// Function: sub_E82CC0
// Address: 0xe82cc0
//
void __fastcall sub_E82CC0(__int64 a1)
{
  __int64 v2; // rdi
  _BYTE *v3; // rax

  if ( *(_BYTE *)(a1 + 16) )
  {
    v2 = *(_QWORD *)(a1 + 8);
    v3 = *(_BYTE **)(v2 + 32);
    if ( (unsigned __int64)v3 >= *(_QWORD *)(v2 + 24) )
    {
      sub_CB5D20(v2, 62);
    }
    else
    {
      *(_QWORD *)(v2 + 32) = v3 + 1;
      *v3 = 62;
    }
  }
  if ( *(_BYTE *)(a1 + 17) )
    sub_CB6E60(
      *(__int64 **)(a1 + 8),
      *(unsigned int *)(*(_QWORD *)(*(_QWORD *)a1 + 64LL) + 4LL * (unsigned int)--*(_DWORD *)(*(_QWORD *)a1 + 72LL) - 4));
}
