// Function: sub_31A6940
// Address: 0x31a6940
//
__int64 __fastcall sub_31A6940(__int64 a1, _BYTE *a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  unsigned int v5; // ecx
  __int64 v6; // rdx
  _BYTE *v7; // r8
  __int64 v8; // rcx
  __int64 v9; // rax
  int v10; // edx
  __int64 result; // rax
  int v12; // edx
  __int64 v13; // rax
  int v14; // edx
  int v15; // r9d

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
    v12 = 1;
    while ( v7 != (_BYTE *)-4096LL )
    {
      v15 = v12 + 1;
      v5 = (v3 - 1) & (v12 + v5);
      v6 = v4 + 16LL * v5;
      v7 = *(_BYTE **)v6;
      if ( *(_BYTE **)v6 == a2 )
        goto LABEL_4;
      v12 = v15;
    }
LABEL_10:
    v8 = *(_QWORD *)(a1 + 160);
    goto LABEL_11;
  }
LABEL_4:
  v8 = *(_QWORD *)(a1 + 160);
  if ( v6 != v4 + 16 * v3 )
  {
    v9 = v8 + 88LL * *(unsigned int *)(v6 + 8);
    v10 = *(_DWORD *)(v9 + 32);
    result = v9 + 8;
    if ( (v10 & 0xFFFFFFFD) == 1 )
      return result;
    return 0;
  }
LABEL_11:
  v13 = v8 + 88LL * *(unsigned int *)(a1 + 168);
  v14 = *(_DWORD *)(v13 + 32);
  result = v13 + 8;
  if ( (v14 & 0xFFFFFFFD) != 1 )
    return 0;
  return result;
}
