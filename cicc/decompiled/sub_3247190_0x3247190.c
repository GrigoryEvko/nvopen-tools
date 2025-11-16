// Function: sub_3247190
// Address: 0x3247190
//
unsigned __int64 __fastcall sub_3247190(__int64 a1, __int64 a2, const void *a3, size_t a4)
{
  unsigned __int64 v4; // rax
  int v5; // edx

  v4 = sub_3246F60(a1, a2, a3, a4);
  if ( *(_DWORD *)(v4 + 24) == -1 )
  {
    v5 = *(_DWORD *)(a1 + 56);
    *(_DWORD *)(a1 + 56) = v5 + 1;
    *(_DWORD *)(v4 + 24) = v5;
  }
  return v4 & 0xFFFFFFFFFFFFFFFBLL;
}
