// Function: sub_AE4640
// Address: 0xae4640
//
bool __fastcall sub_AE4640(__int64 a1, __int64 a2)
{
  char v3; // al
  char v4; // al
  size_t v5; // rdx
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 v9; // r15

  if ( (*(_QWORD *)a1 & 0xFFFFFFFF000000FFLL) != (*(_QWORD *)a2 & 0xFFFFFFFF000000FFLL) )
    return 0;
  if ( *(_QWORD *)(a1 + 8) != *(_QWORD *)(a2 + 8) )
    return 0;
  v3 = *(_BYTE *)(a2 + 17);
  if ( *(_BYTE *)(a1 + 17) )
  {
    if ( *(_BYTE *)(a2 + 16) != *(_BYTE *)(a1 + 16) || !v3 )
      return 0;
  }
  else if ( v3 )
  {
    return 0;
  }
  v4 = *(_BYTE *)(a2 + 19);
  if ( !*(_BYTE *)(a1 + 19) )
  {
    if ( !v4 )
      goto LABEL_11;
    return 0;
  }
  if ( *(_BYTE *)(a1 + 18) != *(_BYTE *)(a2 + 18) || !v4 )
    return 0;
LABEL_11:
  if ( *(_DWORD *)(a1 + 20) != *(_DWORD *)(a2 + 20) )
    return 0;
  if ( *(_DWORD *)(a1 + 24) != *(_DWORD *)(a2 + 24) )
    return 0;
  v5 = *(_QWORD *)(a1 + 40);
  if ( v5 != *(_QWORD *)(a2 + 40) || v5 && memcmp(*(const void **)(a1 + 32), *(const void **)(a2 + 32), v5) )
    return 0;
  if ( !(unsigned __int8)sub_AE45D0(a1 + 64, a2 + 64) )
    return 0;
  if ( !(unsigned __int8)sub_AE45D0(a1 + 128, a2 + 128) )
    return 0;
  if ( !(unsigned __int8)sub_AE45D0(a1 + 176, a2 + 176) )
    return 0;
  v6 = *(unsigned int *)(a1 + 280);
  if ( v6 != *(_DWORD *)(a2 + 280) )
    return 0;
  v7 = *(_QWORD *)(a1 + 272);
  v8 = *(_QWORD *)(a2 + 272);
  v9 = v7 + 20 * v6;
  while ( v9 != v7 )
  {
    if ( !sub_AE1D10(v7, v8) )
      return 0;
    v7 += 20;
    v8 += 20;
  }
  if ( *(_BYTE *)(a1 + 480) != *(_BYTE *)(a2 + 480) )
    return 0;
  return *(_BYTE *)(a1 + 481) == *(_BYTE *)(a2 + 481);
}
