// Function: sub_2D28840
// Address: 0x2d28840
//
bool __fastcall sub_2D28840(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // rcx
  __int64 v6; // rdx
  bool result; // al

  v2 = *(unsigned int *)(a1 + 16);
  v3 = *(unsigned int *)(a2 + 16);
  if ( (_DWORD)v2 && (v4 = *(_QWORD *)(a1 + 8), *(_DWORD *)(v4 + 12) < *(_DWORD *)(v4 + 8)) )
  {
    v5 = v4 + 16 * v2 - 16;
    v6 = *(_QWORD *)(a2 + 8) + 16 * v3 - 16;
    result = 0;
    if ( *(_DWORD *)(v6 + 12) == *(_DWORD *)(v5 + 12) )
      return *(_QWORD *)v6 == *(_QWORD *)v5;
  }
  else
  {
    result = 1;
    if ( (_DWORD)v3 )
      return *(_DWORD *)(*(_QWORD *)(a2 + 8) + 12LL) >= *(_DWORD *)(*(_QWORD *)(a2 + 8) + 8LL);
  }
  return result;
}
