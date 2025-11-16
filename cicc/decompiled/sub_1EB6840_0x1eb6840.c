// Function: sub_1EB6840
// Address: 0x1eb6840
//
__int64 __fastcall sub_1EB6840(__int64 a1, unsigned __int16 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rdx
  const void *v7; // r15
  unsigned int v9; // ecx
  __int64 result; // rax
  __int16 v11; // si
  _WORD *v12; // rcx
  unsigned __int16 *v13; // rdx
  unsigned __int16 v14; // r13
  unsigned __int16 *v15; // r12
  _BYTE *v16; // rdi
  unsigned int v17; // ecx
  unsigned int v18; // eax
  __int64 v19; // rsi
  _DWORD *v20; // rdx
  __int64 v21; // rax

  v6 = *(_QWORD *)(a1 + 248);
  if ( !v6 )
    BUG();
  v7 = (const void *)(a1 + 1040);
  v9 = *(_DWORD *)(*(_QWORD *)(v6 + 8) + 24LL * a2 + 16);
  result = v9 & 0xF;
  v11 = (v9 & 0xF) * a2;
  v12 = (_WORD *)(*(_QWORD *)(v6 + 56) + 2LL * (v9 >> 4));
  v13 = v12 + 1;
  v14 = *v12 + v11;
  while ( 1 )
  {
    v15 = v13;
    if ( !v13 )
      return result;
    while ( 1 )
    {
      v16 = (_BYTE *)(*(_QWORD *)(a1 + 1072) + v14);
      v17 = *(_DWORD *)(a1 + 1032);
      v18 = (unsigned __int8)*v16;
      if ( v18 >= v17 )
        goto LABEL_12;
      v19 = *(_QWORD *)(a1 + 1024);
      while ( 1 )
      {
        v20 = (_DWORD *)(v19 + 4LL * v18);
        if ( v14 == *v20 )
          break;
        v18 += 256;
        if ( v17 <= v18 )
          goto LABEL_12;
      }
      if ( v20 == (_DWORD *)(v19 + 4LL * v17) )
      {
LABEL_12:
        *v16 = v17;
        v21 = *(unsigned int *)(a1 + 1032);
        if ( (unsigned int)v21 >= *(_DWORD *)(a1 + 1036) )
        {
          sub_16CD150(a1 + 1024, v7, 0, 4, a5, a6);
          v21 = *(unsigned int *)(a1 + 1032);
        }
        *(_DWORD *)(*(_QWORD *)(a1 + 1024) + 4 * v21) = v14;
        ++*(_DWORD *)(a1 + 1032);
      }
      result = *v15;
      v13 = 0;
      ++v15;
      v14 += result;
      if ( !(_WORD)result )
        break;
      if ( !v15 )
        return result;
    }
  }
}
