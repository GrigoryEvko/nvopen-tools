// Function: sub_7FAED0
// Address: 0x7faed0
//
void __fastcall sub_7FAED0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 *v3; // rax

  if ( (*(_BYTE *)a1 & 2) == 0 )
  {
    v2 = *(_QWORD *)(a1 + 16);
    if ( v2 )
    {
      v3 = sub_7E5340(v2);
      if ( (*(_BYTE *)a1 & 4) != 0 )
      {
        if ( !v3 )
          return;
        v3 = (__int64 *)*v3;
      }
      if ( v3 )
      {
        if ( *v3 )
          *(_BYTE *)a1 |= 2u;
      }
    }
  }
}
