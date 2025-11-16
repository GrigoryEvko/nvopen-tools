// Function: sub_89A1C0
// Address: 0x89a1c0
//
void __fastcall sub_89A1C0(__int64 *a1, __int64 **a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx

  v2 = **a2;
  *a2 = (__int64 *)v2;
  if ( a1 )
  {
    v3 = *a1;
    if ( *a1 )
    {
      if ( (*(_BYTE *)(v3 + 56) & 0x50) != 0x10 || !v2 || (*(_BYTE *)(v2 + 24) & 8) == 0 )
        *a1 = *(_QWORD *)v3;
    }
  }
  JUMPOUT(0x88F9B1);
}
