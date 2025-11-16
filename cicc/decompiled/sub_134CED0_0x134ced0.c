// Function: sub_134CED0
// Address: 0x134ced0
//
__int64 __fastcall sub_134CED0(__int64 a1, unsigned __int8 a2)
{
  _QWORD *v2; // rdx
  __int64 v4; // rdx
  __int64 v5; // rdx
  void *v6; // rdx

  if ( a2 == 2 )
  {
    v6 = *(void **)(a1 + 24);
    if ( *(_QWORD *)(a1 + 16) - (_QWORD)v6 <= 0xBu )
    {
      sub_16E7EE0(a1, "PartialAlias", 12);
    }
    else
    {
      qmemcpy(v6, "PartialAlias", 12);
      *(_QWORD *)(a1 + 24) += 12LL;
    }
  }
  else if ( a2 > 2u )
  {
    if ( a2 == 3 )
    {
      v4 = *(_QWORD *)(a1 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a1 + 16) - v4) <= 8 )
      {
        sub_16E7EE0(a1, "MustAlias", 9);
      }
      else
      {
        *(_BYTE *)(v4 + 8) = 115;
        *(_QWORD *)v4 = 0x61696C417473754DLL;
        *(_QWORD *)(a1 + 24) += 9LL;
      }
    }
  }
  else if ( a2 )
  {
    v2 = *(_QWORD **)(a1 + 24);
    if ( *(_QWORD *)(a1 + 16) - (_QWORD)v2 <= 7u )
    {
      sub_16E7EE0(a1, "MayAlias", 8);
    }
    else
    {
      *v2 = 0x7361696C4179614DLL;
      *(_QWORD *)(a1 + 24) += 8LL;
    }
  }
  else
  {
    v5 = *(_QWORD *)(a1 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a1 + 16) - v5) <= 6 )
    {
      sub_16E7EE0(a1, "NoAlias", 7);
    }
    else
    {
      *(_DWORD *)v5 = 1816227662;
      *(_WORD *)(v5 + 4) = 24937;
      *(_BYTE *)(v5 + 6) = 115;
      *(_QWORD *)(a1 + 24) += 7LL;
    }
  }
  return a1;
}
