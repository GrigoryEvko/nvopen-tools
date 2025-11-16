// Function: sub_1A23310
// Address: 0x1a23310
//
__int64 **__fastcall sub_1A23310(__int64 a1, __int64 a2)
{
  __int64 **result; // rax
  __int64 *v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  int v9; // r8d
  __int64 v10; // r9
  unsigned __int64 v11; // r15
  unsigned __int64 v12; // r14
  unsigned __int64 v13; // rax
  unsigned __int8 v14; // cl
  char v15; // r8
  int v16; // [rsp+8h] [rbp-38h]
  __int64 v17; // [rsp+8h] [rbp-38h]

  result = *(__int64 ***)(a1 + 336);
  v3 = *(__int64 **)(a2 - 48);
  if ( v3 == *result )
  {
    result = (__int64 **)(a2 | *(_QWORD *)(a1 + 16) & 3LL | 4);
    *(_QWORD *)(a1 + 16) = result;
    goto LABEL_3;
  }
  if ( !*(_BYTE *)(a1 + 344) )
  {
LABEL_3:
    *(_QWORD *)(a1 + 8) = *(_QWORD *)(a1 + 8) & 3LL | a2 | 4;
    return result;
  }
  v4 = sub_15F2050(a2);
  v5 = sub_1632FA0(v4);
  v6 = sub_127FA20(v5, *v3);
  v11 = *(_QWORD *)(a1 + 368);
  v12 = (unsigned __int64)(v6 + 7) >> 3;
  if ( v11 < v12 )
    return (__int64 **)sub_1A21B40(a1, a2, v7, v8, v9, v10);
  v7 = *(unsigned int *)(a1 + 360);
  v10 = a1 + 352;
  if ( (unsigned int)v7 <= 0x40 )
  {
    v13 = *(_QWORD *)(a1 + 352);
  }
  else
  {
    v16 = *(_DWORD *)(a1 + 360);
    v7 = v16 - (unsigned int)sub_16A57B0(a1 + 352);
    if ( (unsigned int)v7 > 0x40 )
      return (__int64 **)sub_1A21B40(a1, a2, v7, v8, v9, v10);
    v10 = a1 + 352;
    v13 = **(_QWORD **)(a1 + 352);
  }
  v17 = v10;
  if ( v11 - v12 < v13 )
    return (__int64 **)sub_1A21B40(a1, a2, v7, v8, v9, v10);
  if ( sub_15F32D0(a2) )
  {
    v15 = *(_BYTE *)(a2 + 18) & 1;
    v14 = *(_BYTE *)(*v3 + 8);
  }
  else
  {
    result = (__int64 **)*v3;
    v14 = *(_BYTE *)(*v3 + 8);
    v15 = *(_BYTE *)(a2 + 18) & 1;
    if ( !v15 )
    {
      if ( v14 > 0x10u )
        goto LABEL_3;
      v15 = ((0x18A7EuLL >> v14) & 1) == 0;
      if ( ((0x18A7EuLL >> v14) & 1) == 0 )
        goto LABEL_3;
    }
  }
  return (__int64 **)sub_1A22CF0((_QWORD *)a1, a2, v17, v12, (((v14 != 11) | v15) ^ 1) & 1, v17);
}
