// Function: sub_131A460
// Address: 0x131a460
//
__int64 __fastcall sub_131A460(struct __pthread_internal_list *a1, unsigned int a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r12d

  if ( pthread_mutex_trylock(&stru_5260DA0) )
  {
    sub_130AD90((__int64)&unk_5260D60);
    unk_5260DC8 = 1;
  }
  ++unk_5260D98;
  if ( a1 != (struct __pthread_internal_list *)unk_5260D90 )
  {
    ++unk_5260D88;
    unk_5260D90 = a1;
  }
  v2 = sub_131A1E0(a1, a2);
  unk_5260DC8 = 0;
  v3 = v2;
  pthread_mutex_unlock(&stru_5260DA0);
  return v3;
}
