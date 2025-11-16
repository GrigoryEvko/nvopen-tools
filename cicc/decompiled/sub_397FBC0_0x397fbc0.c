// Function: sub_397FBC0
// Address: 0x397fbc0
//
__int64 __fastcall sub_397FBC0(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rdx
  __int64 v3; // r8
  int v4; // eax

  while ( 1 )
  {
    if ( *(_BYTE *)a1 != 12 )
      return *(_QWORD *)(a1 + 32);
    if ( (unsigned __int16)(*(_WORD *)(a1 + 2) - 13) > 0x3Au )
      return *(_QWORD *)(a1 + 32);
    v1 = 0x400050002000201LL;
    if ( !_bittest64(&v1, (unsigned int)*(unsigned __int16 *)(a1 + 2) - 13) )
      return *(_QWORD *)(a1 + 32);
    v2 = *(unsigned int *)(a1 + 8);
    v3 = *(_QWORD *)(a1 + 8 * (3 - v2));
    if ( !v3 )
      break;
    v4 = *(unsigned __int16 *)(v3 + 2);
    if ( v4 == 16 || v4 == 66 )
      return *(_QWORD *)(a1 + 32);
    a1 = *(_QWORD *)(a1 + 8 * (3 - v2));
  }
  return 0;
}
