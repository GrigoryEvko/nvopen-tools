// Function: sub_31A6A30
// Address: 0x31a6a30
//
__int64 __fastcall sub_31A6A30(__int64 a1, _BYTE *a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  unsigned int v5; // ecx
  __int64 v6; // rdx
  _BYTE *v7; // r8
  __int64 v8; // rcx
  __int64 v9; // rax
  int v11; // edx
  int v12; // r9d

  if ( !(unsigned __int8)sub_31A68A0(a1, a2) )
    return 0;
  v3 = *(unsigned int *)(a1 + 152);
  v4 = *(_QWORD *)(a1 + 136);
  if ( !(_DWORD)v3 )
    goto LABEL_10;
  v5 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = v4 + 16LL * v5;
  v7 = *(_BYTE **)v6;
  if ( *(_BYTE **)v6 != a2 )
  {
    v11 = 1;
    while ( v7 != (_BYTE *)-4096LL )
    {
      v12 = v11 + 1;
      v5 = (v3 - 1) & (v11 + v5);
      v6 = v4 + 16LL * v5;
      v7 = *(_BYTE **)v6;
      if ( *(_BYTE **)v6 == a2 )
        goto LABEL_4;
      v11 = v12;
    }
LABEL_10:
    v8 = *(_QWORD *)(a1 + 160);
LABEL_11:
    v9 = v8 + 88LL * *(unsigned int *)(a1 + 168);
    if ( *(_DWORD *)(v9 + 32) != 2 )
      return 0;
    return v9 + 8;
  }
LABEL_4:
  v8 = *(_QWORD *)(a1 + 160);
  if ( v6 == v4 + 16 * v3 )
    goto LABEL_11;
  v9 = v8 + 88LL * *(unsigned int *)(v6 + 8);
  if ( *(_DWORD *)(v9 + 32) != 2 )
    return 0;
  return v9 + 8;
}
