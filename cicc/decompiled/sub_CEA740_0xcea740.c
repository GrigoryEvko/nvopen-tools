// Function: sub_CEA740
// Address: 0xcea740
//
char __fastcall sub_CEA740(__int64 a1)
{
  __int64 v2; // rdi
  char result; // al
  __int64 v4; // rax
  unsigned int v5; // r12d
  __int64 v6; // rdx
  _QWORD *v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // rax
  __int64 v10; // rdx
  _QWORD *v11; // rax
  __int64 v12; // rdx
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *v15; // rax
  __int64 v16; // rdx
  _QWORD *v17; // rax

  v2 = *(_QWORD *)(a1 - 32);
  if ( !v2 || *(_BYTE *)v2 || *(_QWORD *)(v2 + 24) != *(_QWORD *)(a1 + 80) )
    return 0;
  result = sub_B2D610(v2, 67);
  if ( !result )
  {
    v4 = *(_QWORD *)(a1 - 32);
    if ( !v4 || *(_BYTE *)v4 || *(_QWORD *)(v4 + 24) != *(_QWORD *)(a1 + 80) || (*(_BYTE *)(v4 + 33) & 0x20) == 0 )
      return 0;
    v5 = *(_DWORD *)(v4 + 36);
    if ( v5 == 8976 )
    {
      v10 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
      v11 = *(_QWORD **)(v10 + 24);
      if ( *(_DWORD *)(v10 + 32) > 0x40u )
        v11 = (_QWORD *)*v11;
      return (((unsigned __int8)v11 >> 2) ^ 1) & 1;
    }
    else
    {
      if ( v5 != 9376 )
      {
        if ( sub_CEA3C0(v5) )
        {
          v6 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
          v7 = *(_QWORD **)(v6 + 24);
          if ( *(_DWORD *)(v6 + 32) > 0x40u )
            v7 = (_QWORD *)*v7;
          return ((BYTE1(v7) >> 5) ^ 1) & 1;
        }
        if ( sub_CEA400(v5) )
        {
          v12 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
          v13 = *(_QWORD **)(v12 + 24);
          if ( *(_DWORD *)(v12 + 32) > 0x40u )
            v13 = (_QWORD *)*v13;
          return ((BYTE4(v13) >> 4) ^ 1) & 1;
        }
        if ( sub_CEA440(v5) )
        {
          v14 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
          v15 = *(_QWORD **)(v14 + 24);
          if ( *(_DWORD *)(v14 + 32) > 0x40u )
            v15 = (_QWORD *)*v15;
          return ((BYTE1(v15) >> 4) ^ 1) & 1;
        }
        if ( sub_CEA470(v5) )
        {
          v16 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
          v17 = *(_QWORD **)(v16 + 24);
          if ( *(_DWORD *)(v16 + 32) > 0x40u )
            v17 = (_QWORD *)*v17;
          return (((unsigned __int8)v17 >> 3) ^ 1) & 1;
        }
        return 0;
      }
      v8 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
      v9 = *(_QWORD **)(v8 + 24);
      if ( *(_DWORD *)(v8 + 32) > 0x40u )
        v9 = (_QWORD *)*v9;
      return (unsigned int)((_DWORD)v9 - 82) > 1;
    }
  }
  return result;
}
