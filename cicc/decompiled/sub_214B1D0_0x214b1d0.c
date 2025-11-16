// Function: sub_214B1D0
// Address: 0x214b1d0
//
__int64 __fastcall sub_214B1D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned __int64 v4; // rax
  __int64 v5; // r13
  unsigned int v6; // r14d
  __int64 v7; // r13
  __int64 v8; // rbx
  unsigned int v9; // eax

  v2 = 100990;
  while ( 1 )
  {
    v4 = *(unsigned __int8 *)(a2 + 8);
    if ( (unsigned __int8)v4 > 0x10u )
      break;
    if ( _bittest64(&v2, v4) )
      return sub_15AAE50(a1, a2);
    if ( (_BYTE)v4 != 14 )
      break;
    a2 = *(_QWORD *)(a2 + 24);
  }
  if ( (_BYTE)v4 == 13 )
  {
    v5 = *(unsigned int *)(a2 + 12);
    v6 = 1;
    if ( (_DWORD)v5 )
    {
      v7 = 8 * v5;
      v8 = 0;
      do
      {
        v9 = sub_214B1D0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 16) + v8), v2);
        if ( v6 < v9 )
          v6 = v9;
        v8 += 8;
      }
      while ( v8 != v7 );
    }
    return v6;
  }
  else
  {
    if ( (_BYTE)v4 != 12 )
      return sub_15AAE50(a1, a2);
    return sub_15A94D0(a1, 0);
  }
}
