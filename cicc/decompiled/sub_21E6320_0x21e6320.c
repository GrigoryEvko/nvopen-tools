// Function: sub_21E6320
// Address: 0x21e6320
//
unsigned __int64 __fastcall sub_21E6320(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v4; // rbx
  unsigned __int64 result; // rax
  _DWORD *v6; // rdx
  _DWORD *v7; // rdx
  __int64 v8; // rdx

  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL * a2 + 8);
  result = (unsigned __int8)v4 >> 4;
  if ( (((unsigned int)v4 >> 4) & 0xF) == 1 )
  {
    v7 = *(_DWORD **)(a3 + 24);
    result = *(_QWORD *)(a3 + 16) - (_QWORD)v7;
    if ( result <= 3 )
    {
      result = sub_16E7EE0(a3, ".cta", 4u);
    }
    else
    {
      *v7 = 1635017518;
      *(_QWORD *)(a3 + 24) += 4LL;
    }
  }
  else if ( (_BYTE)result == 2 )
  {
    v6 = *(_DWORD **)(a3 + 24);
    result = *(_QWORD *)(a3 + 16) - (_QWORD)v6;
    if ( result <= 3 )
    {
      result = sub_16E7EE0(a3, ".sys", 4u);
    }
    else
    {
      *v6 = 1937339182;
      *(_QWORD *)(a3 + 24) += 4LL;
    }
  }
  if ( BYTE2(v4) == 11 )
  {
    v8 = *(_QWORD *)(a3 + 24);
    result = *(_QWORD *)(a3 + 16) - v8;
    if ( result <= 4 )
    {
      return sub_16E7EE0(a3, ".add.", 5u);
    }
    else
    {
      *(_DWORD *)v8 = 1684300078;
      *(_BYTE *)(v8 + 4) = 46;
      *(_QWORD *)(a3 + 24) += 5LL;
    }
  }
  return result;
}
