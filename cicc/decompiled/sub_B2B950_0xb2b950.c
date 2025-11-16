// Function: sub_B2B950
// Address: 0xb2b950
//
void __fastcall sub_B2B950(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rbx
  __int64 v3; // rdi

  v1 = a1 + 72;
  v2 = *(_QWORD *)(a1 + 80);
  *(_BYTE *)(a1 + 128) = 1;
  if ( v2 != a1 + 72 )
  {
    do
    {
      v3 = v2 - 24;
      if ( !v2 )
        v3 = 0;
      sub_AA45D0(v3);
      v2 = *(_QWORD *)(v2 + 8);
    }
    while ( v1 != v2 );
  }
}
