// Function: sub_15A7340
// Address: 0x15a7340
//
__int64 __fastcall sub_15A7340(__int64 a1, __int64 a2)
{
  unsigned int v3; // eax
  _QWORD *v4; // rdi
  __int64 result; // rax

  v3 = *(_DWORD *)(a1 + 104);
  if ( v3 >= *(_DWORD *)(a1 + 108) )
  {
    sub_15A6A10(a1 + 96, 0);
    v3 = *(_DWORD *)(a1 + 104);
  }
  v4 = (_QWORD *)(*(_QWORD *)(a1 + 96) + 8LL * v3);
  if ( v4 )
  {
    *v4 = a2;
    if ( a2 )
      sub_1623A60(v4, a2, 2);
    v3 = *(_DWORD *)(a1 + 104);
  }
  result = v3 + 1;
  *(_DWORD *)(a1 + 104) = result;
  return result;
}
