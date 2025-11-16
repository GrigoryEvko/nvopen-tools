// Function: sub_16A51C0
// Address: 0x16a51c0
//
void __fastcall sub_16A51C0(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax

  if ( a1 != a2 )
  {
    sub_16A5130((unsigned __int64 *)a1, *(_DWORD *)(a2 + 8));
    v2 = *(_DWORD *)(a1 + 8);
    if ( v2 > 0x40 )
      memcpy(*(void **)a1, *(const void **)a2, 8 * (((unsigned __int64)v2 + 63) >> 6));
    else
      *(_QWORD *)a1 = *(_QWORD *)a2;
  }
}
