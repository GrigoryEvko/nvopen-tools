// Function: sub_1FD4DC0
// Address: 0x1fd4dc0
//
bool __fastcall sub_1FD4DC0(__int64 a1, __int64 a2)
{
  int v2; // eax
  int v3; // eax
  __int64 v4; // rdi
  __int64 v6; // rdx
  __int64 v7; // rax

  v2 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v2 > 0x17u )
  {
    if ( (unsigned int)(v2 - 60) <= 0xC
      && sub_15FB940(a2, *(_QWORD *)(a1 + 96))
      && !(unsigned __int8)sub_1FD4DC0(a1, *(_QWORD *)(a2 - 24)) )
    {
      return 0;
    }
    v3 = sub_1FD4C00(a1, a2);
    if ( v3 )
    {
      v6 = *(_QWORD *)(a1 + 56);
      v7 = v3 < 0
         ? *(_QWORD *)(*(_QWORD *)(v6 + 24) + 16LL * (v3 & 0x7FFFFFFF) + 8)
         : *(_QWORD *)(*(_QWORD *)(v6 + 272) + 8LL * (unsigned int)v3);
      if ( v7 )
      {
        if ( (*(_BYTE *)(v7 + 3) & 0x10) == 0 )
          return 0;
        while ( 1 )
        {
          v7 = *(_QWORD *)(v7 + 32);
          if ( !v7 )
            break;
          if ( (*(_BYTE *)(v7 + 3) & 0x10) == 0 )
            return 0;
        }
      }
    }
    if ( *(_BYTE *)(a2 + 16) != 56
      || !(unsigned __int8)sub_15FA1F0(a2)
      || (unsigned __int8)sub_1FD4DC0(a1, *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) )
    {
      v4 = *(_QWORD *)(a2 + 8);
      if ( v4 )
      {
        if ( !*(_QWORD *)(v4 + 8) && (unsigned __int8)(*(_BYTE *)(a2 + 16) - 69) > 2u )
          return *(_QWORD *)(a2 + 40) == sub_1648700(v4)[5];
      }
    }
    return 0;
  }
  return 0;
}
