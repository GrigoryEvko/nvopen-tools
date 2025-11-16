// Function: sub_33E2250
// Address: 0x33e2250
//
__int64 __fastcall sub_33E2250(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // ebx
  __int64 v7; // rax
  void **v8; // rdi
  int v9; // eax
  __int64 result; // rax

  v6 = a4;
  v7 = sub_33E1790(a2, a3, 1u, a4, a5, a6);
  if ( v7 )
  {
    v8 = (void **)(*(_QWORD *)(v7 + 96) + 24LL);
    if ( *v8 == sub_C33340() )
      v9 = sub_C407A0(v8);
    else
      v9 = sub_C35EF0((__int64)v8);
    return v9 >= 0;
  }
  else
  {
    result = 0;
    if ( (unsigned int)(*(_DWORD *)(a2 + 24) - 220) <= 1 )
      return sub_33E0A10(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), v6 + 1);
  }
  return result;
}
