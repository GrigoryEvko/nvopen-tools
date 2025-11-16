// Function: sub_34A0190
// Address: 0x34a0190
//
__int64 __fastcall sub_34A0190(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r14
  _QWORD *v3; // r13
  unsigned int v4; // r13d
  char v5; // al
  unsigned __int64 v7; // rax
  char v8; // al
  int v9; // eax
  __int64 v10; // r9
  __int64 v11; // r14
  __int64 v12; // r15
  unsigned int v13; // eax
  __int64 v14; // [rsp+0h] [rbp-40h]
  __int64 v15; // [rsp+8h] [rbp-38h]

  if ( *(_QWORD *)a1 < *(_QWORD *)a2 )
    return 1;
  v2 = (_QWORD *)(a2 + 8);
  v3 = (_QWORD *)(a1 + 8);
  if ( *(_QWORD *)a1 == *(_QWORD *)a2 )
  {
    v5 = *(_BYTE *)(a1 + 24);
    if ( *(_BYTE *)(a2 + 24) )
    {
      if ( !v5 )
        return 1;
      v7 = *(_QWORD *)(a2 + 8);
      if ( *(_QWORD *)(a1 + 8) < v7 || *(_QWORD *)(a1 + 8) == v7 && *(_QWORD *)(a1 + 16) < *(_QWORD *)(a2 + 16) )
        return 1;
      v5 = sub_34A0170(v2, (_QWORD *)(a1 + 8));
    }
    if ( v5 || *(_QWORD *)(a1 + 32) >= *(_QWORD *)(a2 + 32) )
      goto LABEL_15;
    return 1;
  }
  if ( *(_QWORD *)a1 > *(_QWORD *)a2 )
    return 0;
LABEL_15:
  v8 = *(_BYTE *)(a2 + 24);
  if ( *(_BYTE *)(a1 + 24) )
  {
    if ( !v8 || sub_34A0170(v2, v3) )
      return 0;
    if ( sub_34A0170(v3, v2) )
      goto LABEL_20;
  }
  else if ( v8 )
  {
    goto LABEL_20;
  }
  if ( *(_QWORD *)(a2 + 32) < *(_QWORD *)(a1 + 32) )
    return 0;
LABEL_20:
  v9 = *(_DWORD *)(a2 + 56);
  v4 = 1;
  if ( *(_DWORD *)(a1 + 56) >= v9 )
  {
    v4 = 0;
    if ( *(_DWORD *)(a1 + 56) == v9 )
    {
      v10 = *(_QWORD *)(a1 + 64);
      v11 = *(_QWORD *)(a2 + 64) + 32LL * *(unsigned int *)(a2 + 72);
      v12 = v10 + 32LL * *(unsigned int *)(a1 + 72);
      v14 = *(_QWORD *)(a2 + 64);
      v15 = v10;
      LOBYTE(v13) = sub_349E0F0(v10, v12, v14, v11);
      v4 = v13;
      if ( !(_BYTE)v13 && !sub_349E0F0(v14, v11, v15, v12) )
        LOBYTE(v4) = *(_QWORD *)(a1 + 40) < *(_QWORD *)(a2 + 40);
    }
  }
  return v4;
}
