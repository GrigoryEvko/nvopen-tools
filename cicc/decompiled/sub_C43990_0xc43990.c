// Function: sub_C43990
// Address: 0xc43990
//
void __fastcall sub_C43990(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax

  if ( a1 != a2 )
  {
    sub_C43900((unsigned __int64 *)a1, *(_DWORD *)(a2 + 8));
    v2 = *(_DWORD *)(a1 + 8);
    if ( v2 > 0x40 )
      memcpy(*(void **)a1, *(const void **)a2, 8 * (((unsigned __int64)v2 + 63) >> 6));
    else
      *(_QWORD *)a1 = *(_QWORD *)a2;
  }
}
