// Function: sub_C3E510
// Address: 0xc3e510
//
__int64 __fastcall sub_C3E510(__int64 a1, __int64 a2)
{
  void *v4; // rbx
  void **v5; // rdi
  __int64 v6; // rsi
  __int64 result; // rax
  __int64 v8; // rax

  v4 = sub_C33340();
  do
  {
    v5 = *(void ***)(a1 + 8);
    v6 = *(_QWORD *)(a2 + 8);
    if ( *v5 == v4 )
    {
      result = sub_C3E510(v5, v6);
      if ( (_DWORD)result != 1 )
        return result;
    }
    else
    {
      result = sub_C37950((__int64)v5, v6);
      if ( (_DWORD)result != 1 )
        return result;
    }
    v8 = *(_QWORD *)(a1 + 8);
    a1 = v8 + 24;
    a2 = *(_QWORD *)(a2 + 8) + 24LL;
  }
  while ( v4 == *(void **)(v8 + 24) );
  return sub_C37950(v8 + 24, a2);
}
