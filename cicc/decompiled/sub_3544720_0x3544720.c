// Function: sub_3544720
// Address: 0x3544720
//
__int64 __fastcall sub_3544720(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v6; // rdx
  __int64 v7; // rcx
  _QWORD *v8; // r8
  int v9; // eax
  __int64 v10; // rax
  int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  _QWORD *v15; // r8
  int v16; // eax
  __int64 v17; // rax
  int v18; // eax
  __int64 v19; // rax
  int v20; // eax
  __int64 v21; // rax
  int v22; // eax
  __int64 v23; // rax

  v2 = (*(__int64 *)(a2 + 8) >> 1) & 3;
  if ( v2 == 3 )
  {
    if ( *(_DWORD *)(a2 + 16) == 3 )
      return 0;
  }
  else if ( v2 != 2 )
  {
    return 0;
  }
  if ( *(_DWORD *)(*(_QWORD *)a2 + 200LL) == -1 )
    return 0;
  if ( (_BYTE)qword_503E968 != 1 )
    return 1;
  if ( v2 == 2 )
    return 1;
  v4 = **(_QWORD **)a2;
  v5 = *(_QWORD *)(*(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  if ( sub_2E8B090(v5) || sub_2E8B090(v4) )
    return 1;
  v9 = *(_DWORD *)(v5 + 44);
  if ( (v9 & 4) != 0 || (v9 & 8) == 0 )
  {
    v10 = (*(_QWORD *)(*(_QWORD *)(v5 + 16) + 24LL) >> 21) & 1LL;
  }
  else
  {
    a2 = 0x200000;
    LOBYTE(v10) = sub_2E88A90(v5, 0x200000, 1);
  }
  if ( (_BYTE)v10 && (*(_BYTE *)(v5 + 45) & 0x40) == 0 )
    return 1;
  v11 = *(_DWORD *)(v4 + 44);
  if ( (v11 & 4) != 0 || (v11 & 8) == 0 )
  {
    v12 = (*(_QWORD *)(*(_QWORD *)(v4 + 16) + 24LL) >> 21) & 1LL;
  }
  else
  {
    a2 = 0x200000;
    LOBYTE(v12) = sub_2E88A90(v4, 0x200000, 1);
  }
  if ( (_BYTE)v12 && (*(_BYTE *)(v4 + 45) & 0x40) == 0
    || (unsigned __int8)sub_2E8B100(v5, a2, v6, v7, v8)
    || (unsigned __int8)sub_2E8B100(v4, a2, v13, v14, v15) )
  {
    return 1;
  }
  if ( (unsigned int)*(unsigned __int16 *)(v4 + 68) - 1 > 1 || (*(_BYTE *)(*(_QWORD *)(v4 + 32) + 64LL) & 8) == 0 )
  {
    v16 = *(_DWORD *)(v4 + 44);
    if ( (v16 & 4) != 0 || (v16 & 8) == 0 )
      v17 = (*(_QWORD *)(*(_QWORD *)(v4 + 16) + 24LL) >> 19) & 1LL;
    else
      LOBYTE(v17) = sub_2E88A90(v4, 0x80000, 1);
    if ( !(_BYTE)v17
      && ((unsigned int)*(unsigned __int16 *)(v4 + 68) - 1 > 1 || (*(_BYTE *)(*(_QWORD *)(v4 + 32) + 64LL) & 0x10) == 0) )
    {
      v20 = *(_DWORD *)(v4 + 44);
      if ( (v20 & 4) != 0 || (v20 & 8) == 0 )
        v21 = (*(_QWORD *)(*(_QWORD *)(v4 + 16) + 24LL) >> 20) & 1LL;
      else
        LOBYTE(v21) = sub_2E88A90(v4, 0x100000, 1);
      if ( !(_BYTE)v21 )
        return 0;
    }
  }
  if ( (unsigned int)*(unsigned __int16 *)(v5 + 68) - 1 > 1 || (*(_BYTE *)(*(_QWORD *)(v5 + 32) + 64LL) & 8) == 0 )
  {
    v18 = *(_DWORD *)(v5 + 44);
    if ( (v18 & 4) != 0 || (v18 & 8) == 0 )
      v19 = (*(_QWORD *)(*(_QWORD *)(v5 + 16) + 24LL) >> 19) & 1LL;
    else
      LOBYTE(v19) = sub_2E88A90(v5, 0x80000, 1);
    if ( !(_BYTE)v19
      && ((unsigned int)*(unsigned __int16 *)(v5 + 68) - 1 > 1 || (*(_BYTE *)(*(_QWORD *)(v5 + 32) + 64LL) & 0x10) == 0) )
    {
      v22 = *(_DWORD *)(v5 + 44);
      if ( (v22 & 4) != 0 || (v22 & 8) == 0 )
        v23 = (*(_QWORD *)(*(_QWORD *)(v5 + 16) + 24LL) >> 20) & 1LL;
      else
        LOBYTE(v23) = sub_2E88A90(v5, 0x100000, 1);
      if ( !(_BYTE)v23 )
        return 0;
    }
  }
  return sub_3544350(a1, v4, v5);
}
