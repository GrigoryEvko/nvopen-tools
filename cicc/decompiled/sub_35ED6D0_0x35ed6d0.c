// Function: sub_35ED6D0
// Address: 0x35ed6d0
//
unsigned __int64 __fastcall sub_35ED6D0(unsigned int a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned __int64 result; // rax
  __int64 v4; // rdx
  __int64 v5; // rdx
  __int64 v6; // rdx

  if ( a1 == 3 )
  {
    v6 = *(_QWORD *)(a2 + 32);
    result = *(_QWORD *)(a2 + 24) - v6;
    if ( result <= 2 )
    {
      return sub_CB6200(a2, ".rp", 3u);
    }
    else
    {
      *(_BYTE *)(v6 + 2) = 112;
      *(_WORD *)v6 = 29230;
      *(_QWORD *)(a2 + 32) += 3LL;
    }
  }
  else if ( a1 > 3 )
  {
    if ( a1 != 4 )
      goto LABEL_18;
    v4 = *(_QWORD *)(a2 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v4) <= 2 )
    {
      return sub_CB6200(a2, ".rz", 3u);
    }
    else
    {
      *(_BYTE *)(v4 + 2) = 122;
      *(_WORD *)v4 = 29230;
      *(_QWORD *)(a2 + 32) += 3LL;
      return 29230;
    }
  }
  else
  {
    if ( a1 != 1 )
    {
      if ( a1 == 2 )
      {
        v2 = *(_QWORD *)(a2 + 32);
        result = *(_QWORD *)(a2 + 24) - v2;
        if ( result <= 2 )
          return sub_CB6200(a2, ".rm", 3u);
        *(_BYTE *)(v2 + 2) = 109;
        *(_WORD *)v2 = 29230;
        *(_QWORD *)(a2 + 32) += 3LL;
        return result;
      }
LABEL_18:
      sub_C64ED0("Unexpected rounding mode.", 1u);
    }
    v5 = *(_QWORD *)(a2 + 32);
    result = *(_QWORD *)(a2 + 24) - v5;
    if ( result <= 2 )
    {
      return sub_CB6200(a2, ".rn", 3u);
    }
    else
    {
      *(_BYTE *)(v5 + 2) = 110;
      *(_WORD *)v5 = 29230;
      *(_QWORD *)(a2 + 32) += 3LL;
    }
  }
  return result;
}
