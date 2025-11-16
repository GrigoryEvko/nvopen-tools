// Function: sub_B1A110
// Address: 0xb1a110
//
unsigned __int64 __fastcall sub_B1A110(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r15
  __int64 v5; // r14
  __int64 v6; // rdx
  unsigned int v7; // eax
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  int v11; // edx
  unsigned __int64 result; // rax

  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)(a3 + 40);
  if ( v4 == v5 )
  {
    if ( !(unsigned __int8)sub_B445A0(a2, a3) )
      return a3;
    return a2;
  }
  if ( v5 )
  {
    v6 = (unsigned int)(*(_DWORD *)(v5 + 44) + 1);
    v7 = *(_DWORD *)(v5 + 44) + 1;
  }
  else
  {
    v6 = 0;
    v7 = 0;
  }
  if ( v7 >= *(_DWORD *)(a1 + 32) || !*(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v6) )
    return a2;
  if ( !(unsigned __int8)sub_B192B0(a1, *(_QWORD *)(a2 + 40)) )
    return a3;
  v8 = sub_B192F0(a1, v4, v5);
  if ( v4 == v8 )
    return a2;
  if ( v8 == v5 )
    return a3;
  v9 = v8 + 48;
  v10 = *(_QWORD *)(v8 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v10 == v9 )
    return 0;
  if ( !v10 )
    BUG();
  v11 = *(unsigned __int8 *)(v10 - 24);
  result = v10 - 24;
  if ( (unsigned int)(v11 - 30) >= 0xB )
    return 0;
  return result;
}
