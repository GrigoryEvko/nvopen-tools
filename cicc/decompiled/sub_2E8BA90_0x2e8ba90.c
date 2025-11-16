// Function: sub_2E8BA90
// Address: 0x2e8ba90
//
__int64 __fastcall sub_2E8BA90(__int64 a1, unsigned int a2, unsigned __int64 *a3, __int64 a4)
{
  __int64 v5; // r14
  __int64 result; // rax
  __int64 v7; // r8
  int v8; // eax
  __int64 v9; // rsi
  bool v10; // cl
  bool v11; // dl
  unsigned __int8 v12; // di
  __int64 v13; // rax
  _BYTE *v14; // rax
  unsigned int v15; // eax
  char v16; // bl
  unsigned __int64 v17; // r12
  char v18; // bl
  __int64 v19; // r13
  unsigned __int64 v20; // rax
  __int64 v21; // rsi

  v5 = *(_QWORD *)(a1 + 32) + 40LL * a2;
  if ( *(_BYTE *)v5 )
    return 0;
  v7 = a4;
  if ( (*(_BYTE *)(*(_QWORD *)(a1 + 16) + 24LL) & 2) != 0 || (v15 = sub_2E88F80(a1), v7 = a4, v15 <= a2) )
  {
    v8 = *(_DWORD *)(v5 + 8);
    if ( v8 >= 0 || (v13 = v8 & 0x7FFFFFFF, (unsigned int)v13 >= *(_DWORD *)(v7 + 464)) )
    {
      v9 = 0;
      v10 = 0;
      v11 = 0;
      v12 = 0;
    }
    else
    {
      v14 = (_BYTE *)(*(_QWORD *)(v7 + 456) + 8 * v13);
      v12 = *v14 & 1;
      v10 = (*v14 & 4) != 0;
      v9 = *(_QWORD *)v14 >> 3;
      v11 = (*v14 & 2) != 0;
    }
    return (8 * v9) | (4LL * v10) | v12 | (2LL * v11);
  }
  else
  {
    v16 = *(_BYTE *)(*(_QWORD *)(a1 + 16)
                   + 6 * (*(unsigned __int16 *)(*(_QWORD *)(a1 + 16) + 16LL) + (unsigned __int64)a2)
                   + 8 * (5LL * **(unsigned __int16 **)(a1 + 16) + 5)
                   + 3);
    if ( (unsigned __int8)(v16 - 6) > 5u )
      return sub_2E865D0(a4, *(_DWORD *)(v5 + 8));
    v17 = *a3;
    v18 = v16 - 6;
    v19 = *a3 & 1;
    if ( (*a3 & 1) != 0 )
      v20 = (((v17 >> 1) & ~(-1LL << (v17 >> 58))) >> v18) & 1;
    else
      v20 = (**(_QWORD **)v17 >> v18) & 1LL;
    if ( (_BYTE)v20 )
      return 0;
    result = sub_2E865D0(a4, *(_DWORD *)(v5 + 8));
    if ( (result & 0xFFFFFFFFFFFFFFF9LL) != 0 )
    {
      v21 = 1LL << v18;
      if ( (_BYTE)v19 )
        *a3 = 2 * ((v17 >> 58 << 57) | ~(-1LL << (v17 >> 58)) & (v21 | ~(-1LL << (v17 >> 58)) & (v17 >> 1))) + 1;
      else
        **(_QWORD **)v17 |= v21;
    }
  }
  return result;
}
