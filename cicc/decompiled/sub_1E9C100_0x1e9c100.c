// Function: sub_1E9C100
// Address: 0x1e9c100
//
bool __fastcall sub_1E9C100(__int64 a1, _DWORD *a2, _DWORD *a3)
{
  int v3; // eax
  __int64 v4; // rcx
  __int64 v5; // rax
  _DWORD *v6; // rax
  _DWORD *v7; // rax
  __int64 v9; // rax

  v3 = *(_DWORD *)(a1 + 16);
  v4 = *(_QWORD *)(a1 + 8);
  if ( v3 )
  {
    v9 = (unsigned int)(v3 + 2);
    *(_DWORD *)(a1 + 16) = v9;
    if ( (unsigned int)v9 >= *(_DWORD *)(v4 + 40) )
      return 0;
    v5 = 40 * v9;
  }
  else
  {
    *(_DWORD *)(a1 + 16) = 1;
    v5 = 40;
  }
  v6 = (_DWORD *)(*(_QWORD *)(v4 + 32) + v5);
  *a2 = v6[2];
  LODWORD(v6) = (*v6 >> 8) & 0xFFF;
  a2[1] = (_DWORD)v6;
  if ( !(_DWORD)v6 )
  {
    a3[1] = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL) + 40LL * (unsigned int)(*(_DWORD *)(a1 + 16) + 1) + 24);
    v7 = *(_DWORD **)(*(_QWORD *)(a1 + 8) + 32LL);
    *a3 = v7[2];
    return (*v7 & 0xFFF00) == 0;
  }
  return 0;
}
