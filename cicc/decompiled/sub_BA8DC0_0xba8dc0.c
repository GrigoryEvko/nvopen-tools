// Function: sub_BA8DC0
// Address: 0xba8dc0
//
__int64 __fastcall sub_BA8DC0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // eax
  int v5; // eax
  __int64 v6; // rdx
  __int64 v7; // rax

  v4 = sub_C92610(a2, a3);
  v5 = sub_C92860(a1 + 288, a2, a3, v4);
  if ( v5 == -1 )
    return 0;
  v6 = *(_QWORD *)(a1 + 288);
  v7 = v6 + 8LL * v5;
  if ( v7 == v6 + 8LL * *(unsigned int *)(a1 + 296) )
    return 0;
  else
    return *(_QWORD *)(*(_QWORD *)v7 + 8LL);
}
