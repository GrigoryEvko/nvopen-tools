// Function: sub_35F2080
// Address: 0x35f2080
//
unsigned __int64 __fastcall sub_35F2080(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  unsigned __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rdx

  v4 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
  if ( (_DWORD)v4 == 3 )
  {
    v9 = *(_QWORD *)(a4 + 32);
    result = *(_QWORD *)(a4 + 24) - v9;
    if ( result <= 2 )
    {
      return sub_CB6200(a4, ".rp", 3u);
    }
    else
    {
      *(_BYTE *)(v9 + 2) = 112;
      *(_WORD *)v9 = 29230;
      *(_QWORD *)(a4 + 32) += 3LL;
    }
  }
  else if ( (unsigned int)v4 > 3 )
  {
    if ( (_DWORD)v4 != 4 )
      goto LABEL_18;
    v7 = *(_QWORD *)(a4 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v7) <= 2 )
    {
      return sub_CB6200(a4, ".rz", 3u);
    }
    else
    {
      *(_BYTE *)(v7 + 2) = 122;
      *(_WORD *)v7 = 29230;
      *(_QWORD *)(a4 + 32) += 3LL;
      return 29230;
    }
  }
  else
  {
    if ( (_DWORD)v4 != 1 )
    {
      if ( (_DWORD)v4 == 2 )
      {
        v5 = *(_QWORD *)(a4 + 32);
        result = *(_QWORD *)(a4 + 24) - v5;
        if ( result <= 2 )
          return sub_CB6200(a4, ".rm", 3u);
        *(_BYTE *)(v5 + 2) = 109;
        *(_WORD *)v5 = 29230;
        *(_QWORD *)(a4 + 32) += 3LL;
        return result;
      }
LABEL_18:
      sub_C64ED0("Unexpected rounding mode.", 1u);
    }
    v8 = *(_QWORD *)(a4 + 32);
    result = *(_QWORD *)(a4 + 24) - v8;
    if ( result <= 2 )
    {
      return sub_CB6200(a4, ".rn", 3u);
    }
    else
    {
      *(_BYTE *)(v8 + 2) = 110;
      *(_WORD *)v8 = 29230;
      *(_QWORD *)(a4 + 32) += 3LL;
    }
  }
  return result;
}
