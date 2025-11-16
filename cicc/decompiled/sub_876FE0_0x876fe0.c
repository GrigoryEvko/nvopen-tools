// Function: sub_876FE0
// Address: 0x876fe0
//
__int64 __fastcall sub_876FE0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *v3; // rax
  __int64 v5; // rdx

  v2 = *(_QWORD *)(a2 + 208);
  if ( v2 == a1 )
    return 1;
  if ( v2 )
  {
    if ( a1 )
    {
      if ( dword_4F07588 )
      {
        v5 = *(_QWORD *)(v2 + 32);
        if ( *(_QWORD *)(a1 + 32) == v5 )
        {
          if ( v5 )
            return 1;
        }
      }
    }
  }
  v3 = *(_QWORD **)(*(_QWORD *)(v2 + 168) + 128LL);
  if ( v3 )
  {
    while ( a1 != v3[1] )
    {
      v3 = (_QWORD *)*v3;
      if ( !v3 )
        return 0;
    }
    return 1;
  }
  return 0;
}
