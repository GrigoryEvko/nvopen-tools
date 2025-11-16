// Function: sub_3567D90
// Address: 0x3567d90
//
__int64 __fastcall sub_3567D90(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdx
  unsigned int v5; // eax
  __int64 v6; // r14
  unsigned __int64 v7; // r13

  v3 = a1[3];
  if ( a2 )
  {
    v4 = (unsigned int)(*(_DWORD *)(a2 + 24) + 1);
    v5 = *(_DWORD *)(a2 + 24) + 1;
  }
  else
  {
    v4 = 0;
    v5 = 0;
  }
  if ( v5 >= *(_DWORD *)(v3 + 32) || !*(_QWORD *)(*(_QWORD *)(v3 + 24) + 8 * v4) )
    return 0;
  v6 = a1[4];
  if ( !v6 )
    return 1;
  v7 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !(unsigned __int8)sub_2E6D360(v3, v7, a2) )
    return 0;
  if ( (unsigned __int8)sub_2E6D360(a1[3], v6, a2) )
    return (unsigned int)sub_2E6D360(a1[3], v7, v6) ^ 1;
  else
    return 1;
}
