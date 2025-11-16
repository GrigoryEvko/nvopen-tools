// Function: sub_223FC70
// Address: 0x223fc70
//
__int64 __fastcall sub_223FC70(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v5; // r12
  int v6; // eax
  int v7; // edi
  int v8; // ecx
  __int64 v9; // rdx
  unsigned __int64 v11; // r9

  v5 = a2;
  v6 = a4 & *(_DWORD *)(a1 + 64);
  v7 = *(_DWORD *)(a1 + 64) & 8;
  v8 = v7 & a4;
  if ( v8 )
    v9 = *(_QWORD *)(a1 + 8);
  else
    v9 = *(_QWORD *)(a1 + 32);
  if ( a2 && !v9 || (v6 & 0x18) == 0 )
    return -1;
  v11 = *(_QWORD *)(a1 + 40);
  if ( v11 && v11 > *(_QWORD *)(a1 + 24) )
  {
    if ( !v7 )
    {
      *(_QWORD *)(a1 + 8) = v11;
      *(_QWORD *)(a1 + 16) = v11;
    }
    *(_QWORD *)(a1 + 24) = v11;
  }
  if ( a2 < 0 || *(_QWORD *)(a1 + 24) - v9 < a2 )
    return -1;
  if ( v8 )
    *(_QWORD *)(a1 + 16) = a2 + *(_QWORD *)(a1 + 8);
  if ( (v6 & 0x10) != 0 )
    sub_223FAE0((_QWORD *)a1, *(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 48), a2);
  return v5;
}
