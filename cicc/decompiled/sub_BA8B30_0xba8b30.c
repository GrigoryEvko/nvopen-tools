// Function: sub_BA8B30
// Address: 0xba8b30
//
__int64 __fastcall sub_BA8B30(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  int v5; // eax
  __int64 v6; // rcx
  unsigned int v7; // eax
  int v8; // eax
  __int64 v9; // rax

  v3 = a3;
  v4 = *(_QWORD *)(a1 + 120);
  v5 = *(_DWORD *)(v4 + 24);
  if ( v5 >= 0 )
  {
    v6 = v5;
    if ( v5 < a3 )
    {
      v3 = 1;
      if ( v5 > 1 )
      {
        if ( a3 <= v5 )
          v6 = a3;
        v3 = v6;
      }
    }
  }
  v7 = sub_C92610(a2, v3);
  v8 = sub_C92860(v4, a2, v3, v7);
  if ( v8 == -1 )
    return 0;
  v9 = *(_QWORD *)v4 + 8LL * v8;
  if ( v9 == *(_QWORD *)v4 + 8LL * *(unsigned int *)(v4 + 8) )
    return 0;
  else
    return *(_QWORD *)(*(_QWORD *)v9 + 8LL);
}
