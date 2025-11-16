// Function: sub_1ED9B40
// Address: 0x1ed9b40
//
__int64 __fastcall sub_1ED9B40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v6; // r15
  int v7; // r13d
  unsigned __int64 v8; // r12
  __int64 v9; // rbx
  __int64 *v10; // rax
  __int64 result; // rax
  __int64 *v12; // rdx
  __int64 v13; // rsi
  unsigned int v14; // edi
  unsigned int v15; // ecx
  __int64 v16; // rax

  v6 = *(_QWORD *)(a2 + 104);
  v7 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 256) + 248LL) + 4LL * a5);
  if ( (*(_BYTE *)(a4 + 3) & 0x10) != 0 )
    v7 = ~v7;
  if ( !v6 )
  {
    v8 = a3 & 0xFFFFFFFFFFFFFFF8LL;
LABEL_10:
    *(_BYTE *)(a4 + 4) |= 1u;
    v12 = (__int64 *)sub_1DB3C70((__int64 *)a2, v8);
    v13 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
    if ( v12 == (__int64 *)v13 )
      goto LABEL_16;
    v14 = *(_DWORD *)(v8 + 24);
    v15 = *(_DWORD *)((*v12 & 0xFFFFFFFFFFFFFFF8LL) + 24);
    if ( (unsigned __int64)(v15 | (*v12 >> 1) & 3) <= v14 && v8 == (v12[1] & 0xFFFFFFFFFFFFFFF8LL) )
    {
      if ( (__int64 *)v13 == v12 + 3 )
        goto LABEL_16;
      v16 = v12[3];
      v12 += 3;
      v15 = *(_DWORD *)((v16 & 0xFFFFFFFFFFFFFFF8LL) + 24);
    }
    if ( v15 <= v14 )
    {
      result = v12[1] ^ 6;
      if ( (result & 6) != 0 )
      {
        if ( v12[2] )
          return result;
      }
    }
LABEL_16:
    *(_BYTE *)(a1 + 396) = 1;
    return a1;
  }
  v8 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  v9 = (a3 >> 1) & 3;
  while ( 1 )
  {
    if ( (*(_DWORD *)(v6 + 112) & v7) != 0 )
    {
      v10 = (__int64 *)sub_1DB3C70((__int64 *)v6, a3);
      if ( v10 != (__int64 *)(*(_QWORD *)v6 + 24LL * *(unsigned int *)(v6 + 8)) )
      {
        result = *(_DWORD *)((*v10 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v10 >> 1) & 3;
        if ( (unsigned int)result <= ((unsigned int)v9 | *(_DWORD *)(v8 + 24)) )
          return result;
      }
    }
    v6 = *(_QWORD *)(v6 + 104);
    if ( !v6 )
      goto LABEL_10;
  }
}
