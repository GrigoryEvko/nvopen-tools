// Function: sub_E29250
// Address: 0xe29250
//
unsigned __int64 __fastcall sub_E29250(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 result; // rax
  __int64 v5; // rdx
  char v6; // al

  if ( *(_QWORD *)a2 )
  {
    if ( (unsigned int)(**(char **)(a2 + 8) - 48) <= 4 )
    {
      v6 = sub_E22E20(a1, (_QWORD *)a2);
      return sub_E290B0(a1, (__int64 *)a2, v6);
    }
    else
    {
      result = sub_E28950(a1, (size_t *)a2);
      v5 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a3 + 16) + 16LL) + 8LL * *(_QWORD *)(*(_QWORD *)(a3 + 16) + 24LL) - 8);
      if ( *(_DWORD *)(v5 + 8) == 9 )
      {
        if ( result )
          *(_QWORD *)(v5 + 24) = *(_QWORD *)(*(_QWORD *)(result + 24) + 32LL);
      }
    }
  }
  else
  {
    *(_BYTE *)(a1 + 8) = 1;
    return 0;
  }
  return result;
}
