// Function: sub_1758E80
// Address: 0x1758e80
//
_BOOL8 __fastcall sub_1758E80(__int64 a1, __int64 a2, __int64 *a3, _QWORD *a4, _QWORD *a5, _QWORD *a6, _QWORD *a7)
{
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v10; // rdi
  __int64 v11; // rdi
  int v12; // eax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rcx
  int v17; // edx
  __int64 v18; // rcx

  v7 = *(_QWORD *)(a2 - 48);
  if ( *(_BYTE *)(v7 + 16) != 13 )
    return 0;
  *a6 = v7;
  v8 = *(_QWORD *)(a2 - 72);
  if ( *(_BYTE *)(v8 + 16) != 75 )
    return 0;
  v10 = *(_QWORD *)(v8 - 48);
  if ( !v10 )
    return 0;
  *a3 = v10;
  v11 = *(_QWORD *)(v8 - 24);
  if ( !v11 )
    return 0;
  *a4 = v11;
  v12 = *(unsigned __int16 *)(v8 + 18);
  BYTE1(v12) &= ~0x80u;
  if ( v12 != 32 )
    return 0;
  v13 = *(_QWORD *)(a2 - 24);
  v14 = *a3;
  if ( *(_BYTE *)(v13 + 16) != 79 )
    return 0;
  v15 = *(_QWORD *)(v13 - 72);
  if ( *(_BYTE *)(v15 + 16) != 75 )
    return 0;
  if ( v14 != *(_QWORD *)(v15 - 48) )
    return 0;
  if ( v11 != *(_QWORD *)(v15 - 24) )
    return 0;
  v17 = *(unsigned __int16 *)(v15 + 18);
  v16 = *(_QWORD *)(v13 - 48);
  BYTE1(v17) &= ~0x80u;
  if ( *(_BYTE *)(v16 + 16) != 13 )
    return 0;
  *a5 = v16;
  v18 = *(_QWORD *)(v13 - 24);
  if ( *(_BYTE *)(v18 + 16) != 13 )
    return 0;
  *a7 = v18;
  return v17 == 40;
}
