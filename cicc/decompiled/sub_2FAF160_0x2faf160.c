// Function: sub_2FAF160
// Address: 0x2faf160
//
__int64 __fastcall sub_2FAF160(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  unsigned int v9; // ecx
  _BYTE *v10; // rdi
  unsigned int v11; // eax
  __int64 v12; // rsi
  _DWORD *v13; // rdx
  __int64 result; // rax
  _QWORD *v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  __int64 v19; // rax

  v6 = a2;
  v9 = *(_DWORD *)(a1 + 232);
  v10 = (_BYTE *)(a2 + *(_QWORD *)(a1 + 272));
  v11 = (unsigned __int8)*v10;
  if ( v11 >= v9 )
    goto LABEL_10;
  v12 = *(_QWORD *)(a1 + 224);
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
    v19 = *(unsigned int *)(a1 + 232);
    if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 236) )
    {
      sub_C8D5F0(a1 + 224, (const void *)(a1 + 240), v19 + 1, 4u, a5, a6);
      v19 = *(unsigned int *)(a1 + 232);
    }
    *(_DWORD *)(*(_QWORD *)(a1 + 224) + 4 * v19) = a2;
    ++*(_DWORD *)(a1 + 232);
  }
  result = 1LL << a2;
  v15 = (_QWORD *)(**(_QWORD **)(a1 + 32) + 8LL * (a2 >> 6));
  if ( (*v15 & (1LL << a2)) == 0 )
  {
    *v15 |= result;
    v16 = *(_QWORD *)(a1 + 216);
    v17 = 112 * v6 + *(_QWORD *)(a1 + 24);
    *(_QWORD *)(v17 + 8) = 0;
    *(_QWORD *)v17 = 0;
    *(_DWORD *)(v17 + 16) = 0;
    *(_QWORD *)(v17 + 104) = v16;
    *(_DWORD *)(v17 + 32) = 0;
    result = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL) + 48 * v6;
    if ( *(_DWORD *)(result + 8) > 0x64u )
    {
      *(_QWORD *)(*(_QWORD *)(a1 + 24) + 112 * v6 + 8) = 0;
      v18 = sub_2E3A080(*(_QWORD *)(a1 + 16));
      result = (v18 >> 4 == 0) | (v18 >> 4);
      *(_QWORD *)(*(_QWORD *)(a1 + 24) + 112 * v6) = result;
    }
  }
  return result;
}
