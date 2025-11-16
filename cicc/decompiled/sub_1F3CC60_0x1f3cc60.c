// Function: sub_1F3CC60
// Address: 0x1f3cc60
//
char __fastcall sub_1F3CC60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rcx
  char result; // al
  __int64 v5; // rax

  v3 = *(_QWORD *)(a3 + 8);
  result = 0;
  if ( (unsigned __int64)(v3 + 0xFFFF) <= 0x1FFFD && !*(_QWORD *)a3 )
  {
    v5 = *(_QWORD *)(a3 + 24);
    if ( v5 == 1 )
    {
      return !(*(_BYTE *)(a3 + 16) & (v3 != 0));
    }
    else if ( v5 == 2 )
    {
      return (*(_BYTE *)(a3 + 16) | (v3 != 0)) ^ 1;
    }
    else
    {
      return v5 == 0;
    }
  }
  return result;
}
