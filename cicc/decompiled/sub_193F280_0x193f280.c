// Function: sub_193F280
// Address: 0x193f280
//
bool __fastcall sub_193F280(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  int v5; // edx
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v9; // rdi
  unsigned int v10; // esi
  __int64 v11; // rdx
  int v12; // eax
  __int64 v13; // rcx

  v4 = sub_193E280(a1);
  if ( !v4 )
    return 1;
  v5 = *(unsigned __int16 *)(v4 + 18);
  BYTE1(v5) &= ~0x80u;
  if ( (unsigned int)(v5 - 32) > 1 )
    return 1;
  v6 = *(_QWORD *)(v4 - 24);
  v7 = *(_QWORD *)(v4 - 48);
  if ( *(_BYTE *)(v6 + 16) > 0x17u && !sub_15CC890(a2, *(_QWORD *)(v6 + 40), **(_QWORD **)(a1 + 32)) )
  {
    if ( *(_BYTE *)(v7 + 16) > 0x17u && !sub_15CC890(a2, *(_QWORD *)(v7 + 40), **(_QWORD **)(a1 + 32)) )
      return 1;
    v7 = v6;
  }
  if ( *(_BYTE *)(v7 + 16) != 77 )
  {
    v7 = sub_193F190(v7, a1, a2);
    if ( !v7 )
      return 1;
  }
  v9 = sub_13FCB50(a1);
  v10 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
  if ( !v10 )
    return 1;
  v11 = 24LL * *(unsigned int *)(v7 + 56) + 8;
  v12 = 0;
  while ( 1 )
  {
    v13 = v7 - 24LL * v10;
    if ( (*(_BYTE *)(v7 + 23) & 0x40) != 0 )
      v13 = *(_QWORD *)(v7 - 8);
    if ( v9 == *(_QWORD *)(v13 + v11) )
      break;
    ++v12;
    v11 += 8;
    if ( v10 == v12 )
      return 1;
  }
  if ( v12 < 0 )
    return 1;
  return v7 != sub_193F190(*(_QWORD *)(v13 + 24LL * v12), a1, a2);
}
