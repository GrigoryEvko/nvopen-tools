// Function: sub_AF12D0
// Address: 0xaf12d0
//
void __fastcall sub_AF12D0(__int64 a1, int *a2, const void *a3, __int64 a4)
{
  size_t v6; // rdx
  int v7; // eax

  if ( !*(_BYTE *)(a1 + 20) )
  {
    v6 = *(_QWORD *)(a1 + 8);
    if ( v6 == a4 && (!v6 || !memcmp(*(const void **)a1, a3, v6)) )
    {
      v7 = *a2;
      *(_BYTE *)(a1 + 20) = 1;
      *(_DWORD *)(a1 + 16) = v7;
    }
  }
}
