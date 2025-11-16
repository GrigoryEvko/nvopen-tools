// Function: sub_CC7B70
// Address: 0xcc7b70
//
bool __fastcall sub_CC7B70(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned int v3; // eax
  bool result; // al
  unsigned int v5; // edx
  __int64 v6; // rdx

  *(_QWORD *)a2 = sub_CC78E0(a1);
  *(_QWORD *)(a2 + 8) = v2;
  v3 = *(_DWORD *)(a1 + 44);
  if ( v3 == 5 )
    goto LABEL_5;
  if ( v3 > 5 )
  {
    if ( v3 == 9 )
    {
      if ( *(_DWORD *)a2 )
        return *(_DWORD *)a2 > 9u;
      goto LABEL_5;
    }
    if ( v3 - 27 > 1 )
LABEL_18:
      BUG();
LABEL_5:
    *(_QWORD *)(a2 + 8) = 0;
    *(_QWORD *)a2 = 0x800000040000000ALL;
    return 1;
  }
  if ( v3 != 1 )
    goto LABEL_18;
  v5 = *(_DWORD *)a2;
  if ( !*(_DWORD *)a2 )
  {
    v6 = 4;
    goto LABEL_12;
  }
  result = 0;
  if ( v5 > 3 )
  {
    if ( v5 > 0x13 )
    {
      *(_QWORD *)(a2 + 4) = 0;
      *(_DWORD *)a2 = v5 - 9;
      *(_DWORD *)(a2 + 12) = 0;
      return 1;
    }
    v6 = (v5 - 4) & 0x7FFFFFFF;
LABEL_12:
    *(_QWORD *)(a2 + 8) = 0;
    *(_QWORD *)a2 = (v6 << 32) | 0x800000000000000ALL;
    return 1;
  }
  return result;
}
