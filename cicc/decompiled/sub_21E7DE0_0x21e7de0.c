// Function: sub_21E7DE0
// Address: 0x21e7de0
//
unsigned __int64 __fastcall sub_21E7DE0(__int64 a1, unsigned int a2, __int64 a3)
{
  unsigned __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rdx

  result = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL * a2 + 8);
  if ( (_DWORD)result == 3 )
  {
    v9 = *(_QWORD *)(a3 + 24);
    result = *(_QWORD *)(a3 + 16) - v9;
    if ( result <= 2 )
    {
      return sub_16E7EE0(a3, ".rp", 3u);
    }
    else
    {
      *(_BYTE *)(v9 + 2) = 112;
      *(_WORD *)v9 = 29230;
      *(_QWORD *)(a3 + 24) += 3LL;
    }
  }
  else if ( (unsigned int)result > 3 )
  {
    if ( (_DWORD)result == 4 )
    {
      v7 = *(_QWORD *)(a3 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v7) <= 2 )
      {
        return sub_16E7EE0(a3, ".rz", 3u);
      }
      else
      {
        *(_BYTE *)(v7 + 2) = 122;
        *(_WORD *)v7 = 29230;
        *(_QWORD *)(a3 + 24) += 3LL;
        return 29230;
      }
    }
  }
  else if ( (_DWORD)result == 1 )
  {
    v8 = *(_QWORD *)(a3 + 24);
    result = *(_QWORD *)(a3 + 16) - v8;
    if ( result <= 2 )
    {
      return sub_16E7EE0(a3, ".rn", 3u);
    }
    else
    {
      *(_BYTE *)(v8 + 2) = 110;
      *(_WORD *)v8 = 29230;
      *(_QWORD *)(a3 + 24) += 3LL;
    }
  }
  else if ( (_DWORD)result == 2 )
  {
    v6 = *(_QWORD *)(a3 + 24);
    result = *(_QWORD *)(a3 + 16) - v6;
    if ( result <= 2 )
    {
      return sub_16E7EE0(a3, ".rm", 3u);
    }
    else
    {
      *(_BYTE *)(v6 + 2) = 109;
      *(_WORD *)v6 = 29230;
      *(_QWORD *)(a3 + 24) += 3LL;
    }
  }
  return result;
}
