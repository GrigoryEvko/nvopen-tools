// Function: sub_1F12210
// Address: 0x1f12210
//
unsigned __int64 __fastcall sub_1F12210(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // r13
  unsigned int v9; // ecx
  _BYTE *v10; // rdi
  unsigned int v11; // eax
  __int64 v12; // rsi
  _DWORD *v13; // rdx
  unsigned __int64 result; // rax
  _QWORD *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax

  v6 = a2;
  v9 = *(_DWORD *)(a1 + 472);
  v10 = (_BYTE *)(a2 + *(_QWORD *)(a1 + 512));
  v11 = (unsigned __int8)*v10;
  if ( v11 >= v9 )
    goto LABEL_10;
  v12 = *(_QWORD *)(a1 + 464);
  while ( 1 )
  {
    v13 = (_DWORD *)(v12 + 4LL * v11);
    if ( *v13 == a2 )
      break;
    v11 += 256;
    if ( v9 <= v11 )
      goto LABEL_10;
  }
  if ( v13 == (_DWORD *)(v12 + 4LL * v9) )
  {
LABEL_10:
    *v10 = v9;
    v18 = *(unsigned int *)(a1 + 472);
    if ( (unsigned int)v18 >= *(_DWORD *)(a1 + 476) )
    {
      sub_16CD150(a1 + 464, (const void *)(a1 + 480), 0, 4, a5, a6);
      v18 = *(unsigned int *)(a1 + 472);
    }
    *(_DWORD *)(*(_QWORD *)(a1 + 464) + 4 * v18) = a2;
    ++*(_DWORD *)(a1 + 472);
  }
  result = 1LL << a2;
  v15 = (_QWORD *)(**(_QWORD **)(a1 + 272) + 8LL * (a2 >> 6));
  if ( (*v15 & (1LL << a2)) == 0 )
  {
    *v15 |= result;
    v16 = 112 * v6 + *(_QWORD *)(a1 + 264);
    *(_QWORD *)(v16 + 8) = 0;
    *(_DWORD *)(v16 + 16) = 0;
    *(_QWORD *)v16 = 0;
    v17 = *(_QWORD *)(a1 + 456);
    *(_DWORD *)(v16 + 32) = 0;
    *(_QWORD *)(v16 + 104) = v17;
    result = *(_QWORD *)(*(_QWORD *)(a1 + 240) + 296LL) + 48 * v6;
    if ( *(_DWORD *)(result + 8) > 0x64u )
    {
      *(_QWORD *)(*(_QWORD *)(a1 + 264) + 112 * v6 + 8) = 0;
      result = (unsigned __int64)sub_1DDC5F0(*(_QWORD *)(a1 + 256)) >> 4;
      *(_QWORD *)(*(_QWORD *)(a1 + 264) + 112 * v6) = result;
    }
  }
  return result;
}
