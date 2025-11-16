// Function: sub_37B4200
// Address: 0x37b4200
//
bool __fastcall sub_37B4200(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r15d
  __int64 v7; // r14
  __int64 *v8; // rdx
  unsigned int v9; // eax
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rbx
  bool v15; // cf
  __int64 v16; // rdx
  unsigned int v17; // ecx
  unsigned int v19; // [rsp+8h] [rbp-38h]
  unsigned int v20; // [rsp+Ch] [rbp-34h]
  unsigned int v21; // [rsp+Ch] [rbp-34h]

  if ( (*(_BYTE *)(a2 + 249) & 8) != 0 )
  {
    if ( (*(_BYTE *)(a3 + 249) & 8) == 0 )
      return 0;
  }
  else if ( (*(_BYTE *)(a3 + 249) & 8) != 0 )
  {
    return 1;
  }
  v6 = *(_DWORD *)(a3 + 200);
  v7 = *(unsigned int *)(a2 + 200);
  v8 = *(__int64 **)(*(_QWORD *)a1 + 16LL);
  v9 = *(_DWORD *)(a2 + 200);
  v10 = *v8;
  v11 = *v8 + (v7 << 8);
  if ( (*(_BYTE *)(v11 + 254) & 2) == 0 )
  {
    v21 = *(_DWORD *)(a2 + 200);
    sub_2F8F770(*v8 + (v7 << 8), (_QWORD *)a2, (__int64)v8, a4, a5, a6);
    v9 = v21;
    v10 = **(_QWORD **)(*(_QWORD *)a1 + 16LL);
  }
  v12 = *(unsigned int *)(v11 + 244);
  v13 = (unsigned __int64)v6 << 8;
  v14 = v13 + v10;
  if ( (*(_BYTE *)(v14 + 254) & 2) != 0 )
  {
    v15 = *(_DWORD *)(v14 + 244) < (unsigned int)v12;
    if ( *(_DWORD *)(v14 + 244) > (unsigned int)v12 )
      return 1;
  }
  else
  {
    v19 = *(_DWORD *)(v11 + 244);
    v20 = v9;
    sub_2F8F770(v14, (_QWORD *)a2, v12, v13, a5, a6);
    v9 = v20;
    v15 = *(_DWORD *)(v14 + 244) < v19;
    if ( *(_DWORD *)(v14 + 244) > v19 )
      return 1;
  }
  if ( !v15 )
  {
    v16 = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
    v17 = *(_DWORD *)(v16 + 4LL * v6);
    if ( *(_DWORD *)(v16 + 4 * v7) >= v17 )
    {
      if ( *(_DWORD *)(v16 + 4 * v7) <= v17 )
        return v9 < v6;
      return 0;
    }
    return 1;
  }
  return 0;
}
