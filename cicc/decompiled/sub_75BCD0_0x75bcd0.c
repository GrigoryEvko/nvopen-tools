// Function: sub_75BCD0
// Address: 0x75bcd0
//
unsigned __int64 __fastcall sub_75BCD0(__int64 a1)
{
  unsigned __int64 result; // rax
  __int64 v3; // rdx
  __int64 *v4; // rdx
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // r14

  result = (unsigned int)dword_4F08010;
  while ( 1 )
  {
    while ( (_DWORD)result && (*(_BYTE *)(a1 - 8) & 2) == 0 )
    {
      v4 = *(__int64 **)(a1 + 32);
      if ( !v4 )
        return result;
      v5 = *v4;
      if ( a1 == v5 || (*(_BYTE *)(v5 - 8) & 2) == 0 )
        return result;
      a1 = v5;
    }
    result = *(unsigned __int8 *)(a1 + 203);
    if ( (result & 8) != 0 )
      break;
    *(_BYTE *)(a1 + 203) = result | 8;
    if ( (*(_BYTE *)(a1 + 193) & 0x20) != 0 )
    {
      if ( *(_DWORD *)(a1 + 160) )
      {
        v6 = sub_72B840(a1);
        v7 = v6;
        if ( (*(_BYTE *)(v6 + 29) & 1) != 0 )
        {
          v8 = qword_4F04C50;
          qword_4F04C50 = v6;
          sub_75B260(v6, 0x17u);
          if ( dword_4F077C4 == 2 && (unk_4D048F8 || (*(_BYTE *)(v7 - 8) & 2) != 0) )
            sub_75BE40(v7);
          qword_4F04C50 = v8;
        }
      }
    }
    if ( *(_QWORD *)(a1 + 272) )
      sub_75BCD0();
    if ( *(_QWORD *)(a1 + 320) )
      sub_75BCD0();
    if ( *(_BYTE *)(a1 + 174) == 2 && *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL) + 168LL) + 184LL) )
      sub_75BCD0();
    result = *(_QWORD *)(a1 + 32);
    if ( !result )
      break;
    v3 = *(_QWORD *)result;
    if ( a1 == *(_QWORD *)result || (*(_BYTE *)(v3 - 8) & 2) == 0 )
      break;
    result = (unsigned int)dword_4F08010;
    a1 = v3;
  }
  return result;
}
