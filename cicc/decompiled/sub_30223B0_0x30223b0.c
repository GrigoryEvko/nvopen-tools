// Function: sub_30223B0
// Address: 0x30223b0
//
__int64 __fastcall sub_30223B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  int *v3; // rdx
  int v4; // eax
  int *v5; // rax
  __int64 v6; // rdx
  int *v7; // rdx
  __int64 v8; // r8

  v2 = *(_QWORD *)(a2 + 48);
  v3 = (int *)(v2 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v2 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    return 0;
  v4 = v2 & 7;
  if ( v4 )
  {
    if ( v4 != 3 )
      return 0;
    v5 = v3 + 4;
    v6 = 2LL * *v3;
  }
  else
  {
    *(_QWORD *)(a2 + 48) = v3;
    v5 = (int *)(a2 + 48);
    v6 = 2;
  }
  v7 = &v5[v6];
  if ( v5 == v7 )
    return 0;
  while ( 1 )
  {
    v8 = *(_QWORD *)v5;
    if ( (*(_BYTE *)(*(_QWORD *)v5 + 32LL) & 1) != 0 && *(_DWORD *)(v8 + 80) != 0x7FFFFFFF )
      break;
    v5 += 2;
    if ( v7 == v5 )
      return 0;
  }
  return v8;
}
