// Function: sub_39C0E30
// Address: 0x39c0e30
//
__int64 __fastcall sub_39C0E30(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // r8
  unsigned int v6; // r10d
  int v7; // ebx
  unsigned __int64 v8; // r9
  unsigned __int64 v9; // r9
  unsigned int i; // eax
  __int64 v11; // r9
  unsigned int v12; // eax
  __int64 v13; // rdx
  __int64 result; // rax
  _QWORD *v15; // rdx
  __int64 v16; // rdx

  v4 = *(unsigned int *)(a1 + 24);
  if ( !(_DWORD)v4 )
    return 0;
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (unsigned int)a3 >> 9;
  v7 = 1;
  v8 = (((v6 ^ ((unsigned int)a3 >> 4) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(v6 ^ ((unsigned int)a3 >> 4)) << 32)) >> 22)
     ^ ((v6 ^ ((unsigned int)a3 >> 4) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
      - 1
      - ((unsigned __int64)(v6 ^ ((unsigned int)a3 >> 4)) << 32));
  v9 = ((9 * (((v8 - 1 - (v8 << 13)) >> 8) ^ (v8 - 1 - (v8 << 13)))) >> 15)
     ^ (9 * (((v8 - 1 - (v8 << 13)) >> 8) ^ (v8 - 1 - (v8 << 13))));
  for ( i = (v4 - 1) & (((v9 - 1 - (v9 << 27)) >> 31) ^ (v9 - 1 - ((_DWORD)v9 << 27))); ; i = (v4 - 1) & v12 )
  {
    v11 = v5 + 24LL * i;
    if ( *(_QWORD *)v11 == a2 && *(_QWORD *)(v11 + 8) == a3 )
      break;
    if ( *(_QWORD *)v11 == -8 && *(_QWORD *)(v11 + 8) == -8 )
      return 0;
    v12 = v7 + i;
    ++v7;
  }
  if ( v11 == v5 + 24 * v4 )
    return 0;
  v13 = *(_QWORD *)(a1 + 32) + 96LL * *(unsigned int *)(v11 + 16);
  if ( *(_QWORD *)(a1 + 40) == v13 )
    return 0;
  result = *(unsigned int *)(v13 + 24);
  if ( (_DWORD)result )
  {
    v15 = (_QWORD *)(*(_QWORD *)(v13 + 16) + 16 * result - 16);
    result = 0;
    if ( !v15[1] )
    {
      v16 = *(_QWORD *)(*v15 + 32LL);
      if ( !*(_BYTE *)v16 )
        return *(unsigned int *)(v16 + 8);
    }
  }
  return result;
}
