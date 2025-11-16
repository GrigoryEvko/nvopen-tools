// Function: sub_131A7C0
// Address: 0x131a7c0
//
int __fastcall sub_131A7C0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax

  if ( !a2 || (v2 = sub_130B0E0(a2), v2 > 0x5F5E0FF) )
    LODWORD(v2) = pthread_cond_signal((pthread_cond_t *)(a1 + 8));
  return v2;
}
