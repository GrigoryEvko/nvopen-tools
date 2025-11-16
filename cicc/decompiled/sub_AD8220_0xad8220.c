// Function: sub_AD8220
// Address: 0xad8220
//
bool __fastcall sub_AD8220(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v7; // rdx
  int v8; // eax
  int v9; // r13d
  unsigned int v10; // r14d
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // rbx
  __int64 v16; // rbx
  _BYTE *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  _BYTE *v21; // rbx
  _BYTE *v22; // rbx

  if ( *(_BYTE *)a1 == 18 )
  {
    if ( *(_QWORD *)(a1 + 24) == sub_C33340(a1, a2, a3, a4, a5) )
      v5 = *(_QWORD *)(a1 + 32);
    else
      v5 = a1 + 24;
    return (*(_BYTE *)(v5 + 20) & 7) == 1;
  }
  v7 = *(_QWORD *)(a1 + 8);
  v8 = *(unsigned __int8 *)(v7 + 8);
  if ( (_BYTE)v8 == 17 )
  {
    v9 = *(_DWORD *)(v7 + 32);
    v10 = 0;
    if ( !v9 )
      return 1;
    while ( 1 )
    {
      v11 = sub_AD69F0((unsigned __int8 *)a1, v10);
      v15 = v11;
      if ( !v11 || *(_BYTE *)v11 != 18 )
        break;
      v16 = *(_QWORD *)(v11 + 24) == sub_C33340(a1, v10, v12, v13, v14) ? *(_QWORD *)(v15 + 32) : v15 + 24;
      if ( (*(_BYTE *)(v16 + 20) & 7) != 1 )
        break;
      if ( v9 == ++v10 )
        return 1;
    }
    return 0;
  }
  if ( (unsigned int)(v8 - 17) > 1 )
    return 0;
  v17 = sub_AD7630(a1, 0, v7);
  v21 = v17;
  if ( !v17 || *v17 != 18 )
    return 0;
  if ( *((_QWORD *)v17 + 3) == sub_C33340(a1, 0, v18, v19, v20) )
    v22 = (_BYTE *)*((_QWORD *)v21 + 4);
  else
    v22 = v21 + 24;
  return (v22[20] & 7) == 1;
}
