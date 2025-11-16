// Function: sub_273A7A0
// Address: 0x273a7a0
//
bool __fastcall sub_273A7A0(__int64 *a1, __int64 a2)
{
  unsigned __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // rdx
  __int64 v7; // rdi
  __int64 v8; // rsi
  unsigned int v9; // eax
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rdx
  bool result; // al
  __int64 v14; // rcx

  v4 = sub_2739680(a2);
  v5 = *a1;
  v6 = *(_QWORD *)(v4 + 40);
  v7 = v4;
  if ( v6 )
  {
    v8 = (unsigned int)(*(_DWORD *)(v6 + 44) + 1);
    v9 = *(_DWORD *)(v6 + 44) + 1;
  }
  else
  {
    v8 = 0;
    v9 = 0;
  }
  if ( v9 >= *(_DWORD *)(v5 + 32) )
    return 0;
  v10 = *(_QWORD *)(*(_QWORD *)(v5 + 24) + 8 * v8);
  if ( !v10 )
    return 0;
  if ( *(_DWORD *)(v10 + 72) < *((_DWORD *)a1 + 2) )
    return 0;
  if ( *(_DWORD *)(v10 + 76) > *((_DWORD *)a1 + 3) )
    return 0;
  v11 = a1[2];
  if ( v6 == *(_QWORD *)(v11 + 40) && sub_B445A0(v7, v11) )
    return 0;
  v12 = *(_QWORD *)(a2 + 24);
  if ( *(_BYTE *)v12 == 85 )
  {
    v14 = *(_QWORD *)(v12 - 32);
    result = 1;
    if ( v14 && !*(_BYTE *)v14 && *(_QWORD *)(v14 + 24) == *(_QWORD *)(v12 + 80) && (*(_BYTE *)(v14 + 33) & 0x20) != 0 )
      result = *(_DWORD *)(v14 + 36) != 11;
  }
  else
  {
    result = 1;
  }
  *(_BYTE *)a1[3] |= result;
  return result;
}
