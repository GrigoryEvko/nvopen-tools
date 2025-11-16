// Function: sub_F7D780
// Address: 0xf7d780
//
__int64 __fastcall sub_F7D780(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 result; // rax
  char v4; // dl
  _BYTE *v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rdi

  if ( *(_BYTE *)a2 == 22 )
  {
    v2 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 80LL);
    if ( !v2 )
      BUG();
    for ( result = *(_QWORD *)(v2 + 32); ; result = *(_QWORD *)(result + 8) )
    {
      if ( !result )
        BUG();
      v4 = *(_BYTE *)(result - 24);
      if ( v4 == 78 )
      {
        v5 = *(_BYTE **)(result - 56);
        if ( !v5 )
          BUG();
        if ( *v5 != 22 || (_BYTE *)a2 == v5 )
          return result;
      }
      else
      {
        if ( v4 != 85 )
          return result;
        v6 = *(_QWORD *)(result - 56);
        if ( !v6
          || *(_BYTE *)v6
          || *(_QWORD *)(v6 + 24) != *(_QWORD *)(result + 56)
          || (*(_BYTE *)(v6 + 33) & 0x20) == 0
          || (unsigned int)(*(_DWORD *)(v6 + 36) - 68) > 3 )
        {
          return result;
        }
      }
    }
  }
  if ( *(_BYTE *)a2 <= 0x1Cu )
  {
    v8 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 568) + 72LL) + 80LL);
    if ( v8 )
      v8 -= 24;
    return sub_AA5190(v8);
  }
  else
  {
    v7 = *(_QWORD *)(a1 + 576);
    if ( v7 )
      v7 -= 24;
    return sub_F7D460(a1, a2, v7);
  }
}
