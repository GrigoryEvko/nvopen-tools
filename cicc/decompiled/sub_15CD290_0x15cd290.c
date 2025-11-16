// Function: sub_15CD290
// Address: 0x15cd290
//
bool __fastcall sub_15CD290(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int8 v3; // cl
  __int64 v4; // rdx
  bool result; // al
  __int64 v6; // rcx

  v2 = sub_1648700(a2);
  v3 = *(_BYTE *)(v2 + 16);
  v4 = v2;
  result = 1;
  if ( v3 > 0x17u )
  {
    if ( v3 == 77 )
    {
      if ( (*(_BYTE *)(v4 + 23) & 0x40) != 0 )
        v6 = *(_QWORD *)(v4 - 8);
      else
        v6 = v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
      return sub_15CC510(
               a1,
               *(_QWORD *)(v6
                         + 0xFFFFFFFD55555558LL * (unsigned int)((a2 - v6) >> 3)
                         + 24LL * *(unsigned int *)(v4 + 56)
                         + 8)) != 0;
    }
    else
    {
      return sub_15CC510(a1, *(_QWORD *)(v4 + 40)) != 0;
    }
  }
  return result;
}
