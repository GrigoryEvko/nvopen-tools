// Function: sub_1F1BBC0
// Address: 0x1f1bbc0
//
void __fastcall sub_1F1BBC0(__int64 a1, __int64 a2)
{
  int v3; // ebx
  int v4; // r14d
  int v5; // esi

  if ( (*(_BYTE *)(a2 + 8) & 6) != 0 )
  {
    v3 = 0;
    v4 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 16LL) + 8LL) - *(_DWORD *)(*(_QWORD *)(a1 + 72) + 64LL);
    if ( v4 )
    {
      do
      {
        v5 = v3++;
        sub_1F1B3E0(a1, v5, (int *)a2);
      }
      while ( v4 != v3 );
    }
  }
  else
  {
    sub_1F1B8C0(a1, a2);
  }
}
