// Function: sub_23AE710
// Address: 0x23ae710
//
__int64 __fastcall sub_23AE710(__int64 a1, __int64 a2, const void *a3, size_t a4)
{
  __int64 v6; // r14
  void *v7; // rbx

  v6 = sub_C7D670(a4 + a1 + 1, a2);
  v7 = (void *)(v6 + a1);
  if ( a4 )
    memcpy(v7, a3, a4);
  *((_BYTE *)v7 + a4) = 0;
  return v6;
}
