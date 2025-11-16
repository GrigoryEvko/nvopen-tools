// Function: sub_6011C0
// Address: 0x6011c0
//
__int64 __fastcall sub_6011C0(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax
  __int64 i; // rax
  __int64 v4; // r12
  __int64 v5; // r13
  __int64 *j; // rdx

  if ( (unsigned __int8)(*(_BYTE *)(a1 + 80) - 4) <= 1u && (*(_BYTE *)(*(_QWORD *)(a1 + 88) + 177LL) & 0x30) == 0x30 )
    return 0;
  if ( (*(_BYTE *)(a1 + 81) & 0x10) != 0 )
  {
    for ( i = *(_QWORD *)(a1 + 64); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v4 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)i + 96LL) + 104LL);
    if ( !v4 )
      return 0;
    v5 = *(_QWORD *)(v4 + 88);
    if ( (unsigned int)sub_879510(a1) || (*(_BYTE *)(a1 + 81) & 0x20) != 0 )
    {
      for ( j = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(v5 + 168) + 152LL) + 104LL); j; j = (__int64 *)j[14] )
      {
        while ( 1 )
        {
          result = *j;
          if ( *j )
          {
            if ( *(_BYTE *)(result + 80) == *(_BYTE *)(a1 + 80) )
              break;
          }
          j = (__int64 *)j[14];
          if ( !j )
            goto LABEL_18;
        }
        if ( *(_DWORD *)(*(_QWORD *)(result + 96) + 112LL) == dword_4F06650[0] )
          return result;
      }
    }
    else
    {
      result = sub_883800(*(_QWORD *)(v4 + 96) + 192LL, *(_QWORD *)a1);
      if ( result )
      {
        while ( *(_BYTE *)(result + 80) != *(_BYTE *)(a1 + 80)
             || *(_DWORD *)(*(_QWORD *)(result + 96) + 112LL) != dword_4F06650[0] )
        {
          result = *(_QWORD *)(result + 32);
          if ( !result )
            goto LABEL_18;
        }
        return result;
      }
    }
LABEL_18:
    if ( (unsigned int)sub_8D23B0(*(_QWORD *)(v4 + 88)) )
      result = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
    else
      result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
    if ( !result )
      return 0;
    while ( *(_BYTE *)(result + 80) != *(_BYTE *)(a1 + 80)
         || *(_DWORD *)(*(_QWORD *)(result + 96) + 112LL) != dword_4F06650[0] )
    {
      result = *(_QWORD *)(result + 8);
      if ( !result )
        return 0;
    }
  }
  else
  {
    v1 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 72LL);
    if ( !v1 || (*(_BYTE *)(*(_QWORD *)(a1 + 88) + 178LL) & 1) != 0 )
      return 0;
    return *(_QWORD *)(*(_QWORD *)(v1 + 88) + 176LL);
  }
  return result;
}
