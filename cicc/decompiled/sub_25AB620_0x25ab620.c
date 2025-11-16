// Function: sub_25AB620
// Address: 0x25ab620
//
bool __fastcall sub_25AB620(__int64 a1, __int64 a2)
{
  bool result; // al
  _QWORD *v3; // rdx
  _QWORD *v4; // rcx

  if ( (*(_BYTE *)(a1 + 80) & 1) == 0 )
    return 1;
  if ( sub_B2FC80(a1) )
    return 1;
  if ( (unsigned __int8)sub_B2F6B0(a1) )
    return 1;
  result = (*(_BYTE *)(a1 + 80) & 2) != 0;
  if ( (*(_BYTE *)(a1 + 80) & 2) != 0
    || *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) >> 8
    || (*(_BYTE *)(a1 + 35) & 4) != 0
    || (*(_BYTE *)(a1 + 33) & 0x1C) != 0 )
  {
    return 1;
  }
  if ( !*(_BYTE *)(a2 + 28) )
    return sub_C8CA60(a2, a1) != 0;
  v3 = *(_QWORD **)(a2 + 8);
  v4 = &v3[*(unsigned int *)(a2 + 20)];
  if ( v3 != v4 )
  {
    while ( a1 != *v3 )
    {
      if ( v4 == ++v3 )
        return result;
    }
    return 1;
  }
  return result;
}
