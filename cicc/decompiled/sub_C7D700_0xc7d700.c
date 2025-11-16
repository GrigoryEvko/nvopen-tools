// Function: sub_C7D700
// Address: 0xc7d700
//
int __fastcall sub_C7D700(__int64 a1)
{
  __int64 v1; // rdi

  v1 = a1 + 24;
  if ( *(_QWORD *)(v1 + 8) )
    return posix_madvise(*(void **)(v1 + 8), *(_QWORD *)v1, 4);
  else
    return nullsub_2041();
}
