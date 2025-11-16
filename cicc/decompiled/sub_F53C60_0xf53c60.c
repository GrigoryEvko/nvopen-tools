// Function: sub_F53C60
// Address: 0xf53c60
//
__int64 __fastcall sub_F53C60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  bool v8; // al
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int64 v11; // rcx
  bool v12; // zf
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  __int64 v15; // rax
  unsigned int v16; // edx
  __int64 *v17; // rsi
  __int64 v18; // r13
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // r13
  __int64 v22; // rax

  v6 = *(_QWORD *)(a1 - 32);
  if ( !v6 )
    BUG();
  if ( *(_BYTE *)v6 != 17 )
  {
    sub_F4FB10(a2, a3, a4, a1, a5, a6);
    goto LABEL_13;
  }
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    return 0;
  v8 = sub_B532B0(*(_WORD *)(a1 + 2) & 0x3F);
  v11 = *(unsigned int *)(a3 + 12);
  v12 = !v8;
  v13 = *(unsigned int *)(a3 + 8);
  v14 = v13 + 1;
  if ( v12 )
  {
    if ( v14 > v11 )
    {
      sub_C8D5F0(a3, (const void *)(a3 + 16), v14, 8u, v9, v10);
      v13 = *(unsigned int *)(a3 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a3 + 8 * v13) = 16;
    v15 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
    *(_DWORD *)(a3 + 8) = v15;
    v16 = *(_DWORD *)(v6 + 32);
    v17 = *(__int64 **)(v6 + 24);
    if ( v16 <= 0x40 )
      goto LABEL_8;
  }
  else
  {
    if ( v14 > v11 )
    {
      sub_C8D5F0(a3, (const void *)(a3 + 16), v14, 8u, v9, v10);
      v13 = *(unsigned int *)(a3 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a3 + 8 * v13) = 17;
    v15 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
    *(_DWORD *)(a3 + 8) = v15;
    v16 = *(_DWORD *)(v6 + 32);
    v17 = *(__int64 **)(v6 + 24);
    if ( v16 <= 0x40 )
    {
LABEL_8:
      v18 = 0;
      if ( v16 )
        v18 = (__int64)((_QWORD)v17 << (64 - (unsigned __int8)v16)) >> (64 - (unsigned __int8)v16);
      goto LABEL_10;
    }
  }
  v18 = *v17;
LABEL_10:
  if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, (const void *)(a3 + 16), v15 + 1, 8u, v9, v10);
    v15 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v15) = v18;
  ++*(_DWORD *)(a3 + 8);
LABEL_13:
  v21 = sub_F53C40(*(_WORD *)(a1 + 2) & 0x3F);
  if ( !v21 )
    return 0;
  v22 = *(unsigned int *)(a3 + 8);
  if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, (const void *)(a3 + 16), v22 + 1, 8u, v19, v20);
    v22 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v22) = v21;
  ++*(_DWORD *)(a3 + 8);
  return *(_QWORD *)(a1 - 64);
}
