// Function: sub_27785B0
// Address: 0x27785b0
//
__int64 __fastcall sub_27785B0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdi
  int v3; // eax

  if ( *(_DWORD *)a1 )
  {
    result = 0;
    if ( *(_DWORD *)(a1 + 16) <= 1u )
      return *(unsigned __int8 *)(a1 + 24) ^ 1u;
  }
  else
  {
    v2 = *(_QWORD *)(a1 + 32);
    if ( *(_BYTE *)v2 == 61 || *(_BYTE *)v2 == 62 )
    {
      result = !(*(_WORD *)(v2 + 2) & 1);
      if ( ((*(_WORD *)(v2 + 2) >> 7) & 6) != 0 )
        return 0;
    }
    else
    {
      LOBYTE(v3) = sub_B46500((unsigned __int8 *)v2);
      return v3 ^ 1u;
    }
  }
  return result;
}
