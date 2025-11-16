// Function: sub_18CE100
// Address: 0x18ce100
//
void __fastcall sub_18CE100(__int64 a1)
{
  void *v2; // rdi
  unsigned int v3; // eax
  __int64 v4; // rdx

  ++*(_QWORD *)a1;
  v2 = *(void **)(a1 + 16);
  if ( *(void **)(a1 + 8) == v2 )
    goto LABEL_6;
  v3 = 4 * (*(_DWORD *)(a1 + 28) - *(_DWORD *)(a1 + 32));
  v4 = *(unsigned int *)(a1 + 24);
  if ( v3 < 0x20 )
    v3 = 32;
  if ( (unsigned int)v4 <= v3 )
  {
    memset(v2, -1, 8 * v4);
LABEL_6:
    *(_QWORD *)(a1 + 28) = 0;
    return;
  }
  sub_16CC920(a1);
}
