// Function: sub_2FDDC70
// Address: 0x2fddc70
//
char __fastcall sub_2FDDC70(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  char result; // al
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // r12

  result = sub_2FF7B90(a2);
  if ( result )
  {
    v7 = *(_QWORD *)(a2 + 184);
    result = 0;
    if ( v7 )
    {
      v8 = v7 + 10LL * *(unsigned __int16 *)(*(_QWORD *)(a3 + 16) + 6LL);
      v9 = (unsigned int)*(unsigned __int16 *)(v8 + 6) + a4;
      if ( *(unsigned __int16 *)(v8 + 8) > (unsigned int)v9 )
        return *(_DWORD *)(*(_QWORD *)(a2 + 168) + 4 * v9) <= 1u;
    }
  }
  return result;
}
