// Function: sub_1B916B0
// Address: 0x1b916b0
//
void __fastcall sub_1B916B0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // r12
  __int64 *v5; // rbx

  v4 = &a2[a3];
  if ( v4 != a2 )
  {
    v5 = a2;
    do
    {
      if ( *(_BYTE *)(*v5 + 16) > 0x17u )
        sub_1B91660(a1, *v5, a4);
      ++v5;
    }
    while ( v4 != v5 );
  }
}
