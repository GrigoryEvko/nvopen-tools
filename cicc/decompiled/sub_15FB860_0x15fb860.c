// Function: sub_15FB860
// Address: 0x15fb860
//
bool __fastcall sub_15FB860(_BYTE *a1)
{
  bool result; // al
  __int64 v2; // rdx

  result = 0;
  if ( a1[16] == 71 )
  {
    v2 = **((_QWORD **)a1 - 3);
    if ( v2 == *(_QWORD *)a1 )
    {
      return 1;
    }
    else if ( *(_BYTE *)(v2 + 8) == 15 )
    {
      return *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 15;
    }
  }
  return result;
}
