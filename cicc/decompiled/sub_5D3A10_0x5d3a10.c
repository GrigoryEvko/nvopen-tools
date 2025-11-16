// Function: sub_5D3A10
// Address: 0x5d3a10
//
void __fastcall sub_5D3A10(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx

  if ( *(_BYTE *)(a1 + 24) == 3 && (*(_BYTE *)(a1 + 25) & 1) != 0 )
  {
    v2 = *(_QWORD *)(a1 + 56);
    if ( (unsigned int)sub_8D3410(*(_QWORD *)(v2 + 120)) || (unsigned int)sub_8D3A70(*(_QWORD *)(v2 + 120)) )
    {
      qword_4CF7C70 = v2;
      *(_DWORD *)(a2 + 72) = 1;
    }
  }
}
