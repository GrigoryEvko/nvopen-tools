// Function: sub_1409FE0
// Address: 0x1409fe0
//
void __fastcall sub_1409FE0(__int64 a1)
{
  void *v2; // rdi
  unsigned int v3; // eax
  __int64 v4; // rdx

  ++*(_QWORD *)(a1 + 208);
  *(_DWORD *)(a1 + 168) = 0;
  v2 = *(void **)(a1 + 224);
  if ( v2 == *(void **)(a1 + 216) )
    goto LABEL_6;
  v3 = 4 * (*(_DWORD *)(a1 + 236) - *(_DWORD *)(a1 + 240));
  v4 = *(unsigned int *)(a1 + 232);
  if ( v3 < 0x20 )
    v3 = 32;
  if ( (unsigned int)v4 <= v3 )
  {
    memset(v2, -1, 8 * v4);
LABEL_6:
    *(_QWORD *)(a1 + 236) = 0;
    return;
  }
  sub_16CC920(a1 + 208);
}
