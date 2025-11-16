// Function: sub_1070FC0
// Address: 0x1070fc0
//
__int64 __fastcall sub_1070FC0(__int64 a1, __int64 *a2, __int64 a3)
{
  int v6; // eax
  __int64 v7; // rcx
  int v8; // esi
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r8
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rcx
  int v19; // eax
  int v20; // r9d

  v6 = *(_DWORD *)(a1 + 248);
  v7 = *(_QWORD *)(a1 + 232);
  if ( !v6 )
  {
LABEL_10:
    v12 = 0;
    goto LABEL_4;
  }
  v8 = v6 - 1;
  v9 = (v6 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( a3 != *v10 )
  {
    v19 = 1;
    while ( v11 != -4096 )
    {
      v20 = v19 + 1;
      v9 = v8 & (v19 + v9);
      v10 = (__int64 *)(v7 + 16LL * v9);
      v11 = *v10;
      if ( a3 == *v10 )
        goto LABEL_3;
      v19 = v20;
    }
    goto LABEL_10;
  }
LABEL_3:
  v12 = v10[1];
LABEL_4:
  v13 = sub_E5CAC0(a2, a3);
  v14 = 0;
  v15 = v13;
  v16 = (unsigned int)(*(_DWORD *)(a3 + 172) + 1);
  if ( (unsigned int)v16 < *(_DWORD *)(a1 + 264) )
  {
    v17 = *(_QWORD *)(*(_QWORD *)(a1 + 256) + 8 * v16);
    if ( (*(_BYTE *)(v17 + 48) & 0x20) == 0 )
      return (-(1LL << *(_BYTE *)(v17 + 32)) & (v15 + v12 + (1LL << *(_BYTE *)(v17 + 32)) - 1)) - (v15 + v12);
  }
  return v14;
}
