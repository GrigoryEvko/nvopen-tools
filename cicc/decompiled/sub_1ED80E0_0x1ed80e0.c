// Function: sub_1ED80E0
// Address: 0x1ed80e0
//
void __fastcall sub_1ED80E0(__int64 a1)
{
  void *v2; // rdi
  unsigned int v3; // eax
  __int64 v4; // rdx

  ++*(_QWORD *)(a1 + 560);
  v2 = *(void **)(a1 + 576);
  if ( v2 != *(void **)(a1 + 568) )
  {
    v3 = 4 * (*(_DWORD *)(a1 + 588) - *(_DWORD *)(a1 + 592));
    v4 = *(unsigned int *)(a1 + 584);
    if ( v3 < 0x20 )
      v3 = 32;
    if ( (unsigned int)v4 > v3 )
    {
      sub_16CC920(a1 + 560);
      goto LABEL_7;
    }
    memset(v2, -1, 8 * v4);
  }
  *(_QWORD *)(a1 + 588) = 0;
LABEL_7:
  *(_DWORD *)(a1 + 408) = 0;
  *(_DWORD *)(a1 + 672) = 0;
  *(_DWORD *)(a1 + 752) = 0;
}
