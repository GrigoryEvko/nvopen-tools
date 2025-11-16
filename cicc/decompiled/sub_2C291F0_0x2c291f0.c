// Function: sub_2C291F0
// Address: 0x2c291f0
//
__int64 __fastcall sub_2C291F0(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rsi
  __int64 v3; // rcx
  __int64 v4; // rax
  __int64 v5; // rdx

  v1 = *(_QWORD *)(a1 + 112);
  if ( (unsigned int)*(unsigned __int8 *)(v1 + 8) - 1 > 1 )
    return 0;
  v2 = *(_QWORD *)(v1 + 120);
  v3 = v1 + 112;
  if ( v2 == v1 + 112 )
    return 0;
  v4 = *(_QWORD *)(v1 + 120);
  v5 = 0;
  do
  {
    v4 = *(_QWORD *)(v4 + 8);
    ++v5;
  }
  while ( v3 != v4 );
  if ( v5 != 1 )
    return 0;
  if ( !v2 )
    BUG();
  if ( *(_BYTE *)(v2 - 16) )
    return 0;
  else
    return **(_QWORD **)(v2 + 24);
}
