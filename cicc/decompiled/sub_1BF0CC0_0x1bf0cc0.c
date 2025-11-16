// Function: sub_1BF0CC0
// Address: 0x1bf0cc0
//
bool __fastcall sub_1BF0CC0(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r12
  unsigned __int64 v5; // rax
  __int64 v6; // r9
  __int64 v7; // rax
  char v8; // di
  unsigned int v9; // esi
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rbx
  __int64 v14; // r12
  __int64 v15; // rbx
  __int64 v16; // r14

  if ( a1 == a2 )
    return 1;
  v3 = sub_13FCD20(a1);
  if ( !v3 )
    return 0;
  v4 = sub_13FCB50(a1);
  v5 = sub_157EBA0(v4);
  if ( *(_BYTE *)(v5 + 16) != 26 )
    return 0;
  if ( (*(_DWORD *)(v5 + 20) & 0xFFFFFFF) == 1 )
    return 0;
  v6 = *(_QWORD *)(v5 - 72);
  if ( (unsigned __int8)(*(_BYTE *)(v6 + 16) - 75) > 1u )
    return 0;
  v7 = 0x17FFFFFFE8LL;
  v8 = *(_BYTE *)(v3 + 23) & 0x40;
  v9 = *(_DWORD *)(v3 + 20) & 0xFFFFFFF;
  if ( v9 )
  {
    v10 = 24LL * *(unsigned int *)(v3 + 56) + 8;
    v11 = 0;
    do
    {
      v12 = v3 - 24LL * v9;
      if ( v8 )
        v12 = *(_QWORD *)(v3 - 8);
      if ( v4 == *(_QWORD *)(v12 + v10) )
      {
        v7 = 24 * v11;
        goto LABEL_13;
      }
      ++v11;
      v10 += 8;
    }
    while ( v9 != (_DWORD)v11 );
    v7 = 0x17FFFFFFE8LL;
  }
LABEL_13:
  if ( v8 )
    v13 = *(_QWORD *)(v3 - 8);
  else
    v13 = v3 - 24LL * v9;
  v14 = *(_QWORD *)(v6 - 48);
  v15 = *(_QWORD *)(v13 + v7);
  v16 = *(_QWORD *)(v6 - 24);
  if ( v14 != v15 || !sub_13FC1A0(a2, *(_QWORD *)(v6 - 24)) )
  {
    if ( v16 == v15 )
      return sub_13FC1A0(a2, v14);
    return 0;
  }
  return 1;
}
