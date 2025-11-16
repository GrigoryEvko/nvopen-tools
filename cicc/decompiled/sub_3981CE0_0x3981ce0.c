// Function: sub_3981CE0
// Address: 0x3981ce0
//
__int64 __fastcall sub_3981CE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v7; // al
  __int16 v8; // dx
  _QWORD *v9; // rax
  unsigned __int64 v10; // rbx
  __int64 v11; // rdx
  unsigned __int16 v12; // di
  unsigned int v13; // esi
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r8
  __int64 *v17; // rax
  __int64 i; // rbx
  __int64 v19; // r8
  __int64 *v20; // rax
  __int64 v22; // [rsp+8h] [rbp-38h]
  __int64 v23; // [rsp+8h] [rbp-38h]

  v7 = *(_BYTE *)(a2 + 30);
  if ( !v7 )
    v7 = *(_QWORD *)(a2 + 32) != 0;
  v8 = *(_WORD *)(a2 + 28);
  *(_BYTE *)(a1 + 14) = v7;
  *(_QWORD *)a1 = 0;
  *(_WORD *)(a1 + 12) = v8;
  *(_QWORD *)(a1 + 16) = a1 + 32;
  *(_QWORD *)(a1 + 24) = 0xC00000000LL;
  v9 = *(_QWORD **)(a2 + 8);
  if ( v9 )
  {
    v10 = *v9 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v10 )
    {
      v11 = *(unsigned __int16 *)(v10 + 14);
      v12 = *(_WORD *)(v10 + 12);
      v13 = 12;
      v14 = 0;
      if ( (_WORD)v11 == 33 )
        goto LABEL_11;
LABEL_6:
      v15 = v11 << 16;
      v16 = v15 | v12;
      if ( (unsigned int)v14 >= v13 )
      {
        v22 = v15 | v12;
        sub_16CD150(a1 + 16, (const void *)(a1 + 32), 0, 16, v15 | v12, a6);
        v14 = *(unsigned int *)(a1 + 24);
        v16 = v22;
      }
      v17 = (__int64 *)(*(_QWORD *)(a1 + 16) + 16 * v14);
      *v17 = v16;
      v17[1] = 0;
      ++*(_DWORD *)(a1 + 24);
      for ( i = *(_QWORD *)v10; (i & 4) == 0; i = *(_QWORD *)v10 )
      {
        v10 = i & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v10 )
          break;
        v11 = *(unsigned __int16 *)(v10 + 14);
        v14 = *(unsigned int *)(a1 + 24);
        v13 = *(_DWORD *)(a1 + 28);
        v12 = *(_WORD *)(v10 + 12);
        if ( (_WORD)v11 != 33 )
          goto LABEL_6;
LABEL_11:
        a6 = *(_QWORD *)(v10 + 16);
        v19 = v12 | 0x210000LL;
        if ( (unsigned int)v14 >= v13 )
        {
          v23 = *(_QWORD *)(v10 + 16);
          sub_16CD150(a1 + 16, (const void *)(a1 + 32), 0, 16, v12 | 0x210000, a6);
          v14 = *(unsigned int *)(a1 + 24);
          v19 = v12 | 0x210000LL;
          a6 = v23;
        }
        v20 = (__int64 *)(*(_QWORD *)(a1 + 16) + 16 * v14);
        *v20 = v19;
        v20[1] = a6;
        ++*(_DWORD *)(a1 + 24);
      }
    }
  }
  return a1;
}
