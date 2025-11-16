// Function: sub_35ED5A0
// Address: 0x35ed5a0
//
unsigned __int64 __fastcall sub_35ED5A0(unsigned int a1, __int64 a2)
{
  _DWORD *v2; // rdx
  unsigned __int64 result; // rax
  _QWORD *v4; // rdx
  _DWORD *v5; // rdx
  _DWORD *v6; // rdx

  if ( a1 == 2 )
  {
    v6 = *(_DWORD **)(a2 + 32);
    result = *(_QWORD *)(a2 + 24) - (_QWORD)v6;
    if ( result <= 3 )
    {
      return sub_CB6200(a2, (unsigned __int8 *)".sys", 4u);
    }
    else
    {
      *v6 = 1937339182;
      *(_QWORD *)(a2 + 32) += 4LL;
    }
  }
  else if ( a1 > 2 )
  {
    if ( a1 != 3 )
      BUG();
    v4 = *(_QWORD **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v4 <= 7u )
    {
      return sub_CB6200(a2, (unsigned __int8 *)".cluster", 8u);
    }
    else
    {
      *v4 = 0x72657473756C632ELL;
      *(_QWORD *)(a2 + 32) += 8LL;
      return 0x72657473756C632ELL;
    }
  }
  else if ( a1 )
  {
    v5 = *(_DWORD **)(a2 + 32);
    result = *(_QWORD *)(a2 + 24) - (_QWORD)v5;
    if ( result <= 3 )
    {
      return sub_CB6200(a2, (unsigned __int8 *)".cta", 4u);
    }
    else
    {
      *v5 = 1635017518;
      *(_QWORD *)(a2 + 32) += 4LL;
    }
  }
  else
  {
    v2 = *(_DWORD **)(a2 + 32);
    result = *(_QWORD *)(a2 + 24) - (_QWORD)v2;
    if ( result <= 3 )
    {
      return sub_CB6200(a2, (unsigned __int8 *)".gpu", 4u);
    }
    else
    {
      *v2 = 1970300718;
      *(_QWORD *)(a2 + 32) += 4LL;
    }
  }
  return result;
}
