// Function: sub_16C5A40
// Address: 0x16c5a40
//
int __fastcall sub_16C5A40(__int64 a1)
{
  int result; // eax

  if ( *(_QWORD *)(a1 + 8) )
    return munmap(*(void **)(a1 + 8), *(_QWORD *)a1);
  return result;
}
