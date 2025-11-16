// Function: sub_270F0C0
// Address: 0x270f0c0
//
void __fastcall sub_270F0C0(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  __int64 v3; // rdx

  ++*(_QWORD *)a1;
  if ( *(_BYTE *)(a1 + 28) )
    goto LABEL_6;
  v2 = 4 * (*(_DWORD *)(a1 + 20) - *(_DWORD *)(a1 + 24));
  v3 = *(unsigned int *)(a1 + 16);
  if ( v2 < 0x20 )
    v2 = 32;
  if ( (unsigned int)v3 <= v2 )
  {
    memset(*(void **)(a1 + 8), -1, 8 * v3);
LABEL_6:
    *(_QWORD *)(a1 + 20) = 0;
    return;
  }
  sub_C8C990(a1, a2);
}
