// Function: sub_72E220
// Address: 0x72e220
//
__int64 __fastcall sub_72E220(__int64 a1, __int64 a2)
{
  int i; // r12d
  char v3; // al
  int v4; // ebx
  _BYTE *v5; // rcx
  int v6; // eax
  __int64 v7; // r13
  __int64 v8; // rdx
  char v9; // al
  __int64 v11; // r13
  int v12; // eax
  int v13; // ebx
  int v14; // eax
  int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 *v20; // rdi
  int v21; // r14d
  int v22; // eax

  for ( i = 0; ; i += v4 )
  {
    v3 = *(_BYTE *)(a1 + 89);
    v4 = 0;
    if ( (v3 & 0x40) == 0 )
    {
      v5 = (v3 & 8) != 0 ? *(_BYTE **)(a1 + 24) : *(_BYTE **)(a1 + 8);
      v4 = 0;
      if ( v5 )
      {
        while ( 1 )
        {
          v6 = (char)*v5;
          if ( !*v5 )
            break;
          ++v5;
          v4 += 32 * v4 + v6;
        }
      }
    }
    v7 = *(_QWORD *)(a1 + 48);
    if ( v7 )
    {
      v12 = *(_DWORD *)(v7 + 168);
      if ( !v12 )
      {
        v15 = sub_72E220(*(_QWORD *)(a1 + 48));
        v20 = *(__int64 **)(v7 + 240);
        v21 = v15;
        if ( v20 )
          v21 = sub_72E120(v20, a2, v16, v17, v18, v19) + v15;
        v22 = 1;
        if ( v21 )
          v22 = v21;
        *(_DWORD *)(v7 + 168) = v22;
        v13 = v22 + v4;
        return (unsigned int)(i + v13);
      }
      goto LABEL_17;
    }
    v8 = *(_QWORD *)(a1 + 40);
    if ( !v8 )
      return (unsigned int)(i + v4);
    v9 = *(_BYTE *)(v8 + 28);
    if ( v9 == 6 )
      break;
    if ( v9 != 16 )
    {
      if ( v9 != 3 )
        return (unsigned int)(i + v4);
      v11 = *(_QWORD *)(v8 + 32);
      v12 = *(_DWORD *)(v11 + 120);
      if ( !v12 )
      {
        v14 = sub_72E220(*(_QWORD *)(v8 + 32));
        if ( v14 )
        {
          *(_DWORD *)(v11 + 120) = v14;
          v13 = v14 + v4;
        }
        else
        {
          *(_DWORD *)(v11 + 120) = 1;
          v13 = v4 + 1;
        }
        return (unsigned int)(i + v13);
      }
LABEL_17:
      v13 = v12 + v4;
      return (unsigned int)(i + v13);
    }
    a1 = *(_QWORD *)(v8 + 32);
  }
  v13 = sub_72E370(*(_QWORD *)(v8 + 32)) + v4;
  return (unsigned int)(i + v13);
}
