// Function: sub_C82B90
// Address: 0xc82b90
//
int __fastcall sub_C82B90(__int64 a1)
{
  int result; // eax

  if ( *(_QWORD *)(a1 + 8) )
    return munmap(*(void **)(a1 + 8), *(_QWORD *)a1);
  return result;
}
