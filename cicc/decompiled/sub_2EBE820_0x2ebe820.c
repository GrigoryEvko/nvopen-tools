// Function: sub_2EBE820
// Address: 0x2ebe820
//
__int64 __fastcall sub_2EBE820(__int64 a1, int a2, int a3, __int64 a4)
{
  unsigned __int64 v4; // r8
  bool v5; // r9
  bool v6; // al
  unsigned __int8 v7; // r10
  __int64 v8; // rbx
  bool v9; // r12
  bool v10; // r13
  char v11; // r14
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // r9
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned __int64 v17; // r9
  __int64 *v18; // rax
  __int64 result; // rax
  __int64 v20; // rax
  unsigned __int64 v21; // r8
  char v22; // al
  __int64 v23; // r11
  _BYTE *v24; // r11

  if ( a2 >= 0 || (v20 = a2 & 0x7FFFFFFF, (unsigned int)v20 >= *(_DWORD *)(a1 + 464)) )
  {
    v4 = 0;
    v5 = 0;
    v6 = 0;
    v7 = 0;
    if ( a3 < 0 )
      goto LABEL_17;
LABEL_3:
    v8 = 0;
    v9 = 0;
    v10 = 0;
    v11 = 0;
    goto LABEL_4;
  }
  v21 = *(_QWORD *)(*(_QWORD *)(a1 + 456) + 8 * v20);
  v22 = v21;
  v7 = v21 & 1;
  v5 = (v21 & 4) != 0;
  v4 = v21 >> 3;
  v6 = (v22 & 2) != 0;
  if ( a3 >= 0 )
    goto LABEL_3;
LABEL_17:
  v23 = a3 & 0x7FFFFFFF;
  if ( (unsigned int)v23 >= *(_DWORD *)(a1 + 464) )
    goto LABEL_3;
  v24 = (_BYTE *)(*(_QWORD *)(a1 + 456) + 8 * v23);
  v11 = *v24 & 1;
  v9 = (*v24 & 4) != 0;
  v8 = *(_QWORD *)v24 >> 3;
  v10 = (*v24 & 2) != 0;
LABEL_4:
  v12 = 8 * v4;
  v13 = v12 | (4LL * v5) | v7 | (2LL * v6);
  if ( (v13 & 0xFFFFFFFFFFFFFFF9LL) != 0 )
  {
    v14 = 4LL * v9;
    v12 = (8 * v8) | v14 | (2LL * v10) | v11 & 1;
    if ( (8 * v8) | v14 & 0xFFFFFFFFFFFFFFF9LL | (unsigned __int8)(2 * v10) & 0xF9 | (unsigned __int64)(v11 & 1) )
    {
      if ( (((unsigned __int8)v12 ^ (unsigned __int8)v13) & 7) != 0 || ((v12 ^ v13) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        return 0;
    }
  }
  v15 = *(_QWORD *)(a1 + 56);
  v16 = *(_QWORD *)(v15 + 16LL * (a3 & 0x7FFFFFFF));
  v17 = v16 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v16 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_11;
  v18 = (__int64 *)(16LL * (a2 & 0x7FFFFFFF) + v15);
  v12 = *v18;
  if ( (*v18 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
    *v18 = v16;
    goto LABEL_11;
  }
  if ( !((v16 >> 2) & 1) != !((v12 >> 2) & 1) )
    return 0;
  if ( ((v16 >> 2) & 1) != 0 )
  {
    if ( v12 != v16 )
      return 0;
  }
  else if ( !sub_2EBE500(a1, a2, *v18 & 0xFFFFFFFFFFFFFFF8LL, v17, a4) )
  {
    return 0;
  }
LABEL_11:
  result = 1;
  if ( (8 * v8) | (unsigned __int8)(4 * v9) & 0xF9 | v11 & 1 | (unsigned __int64)((unsigned __int8)(2 * v10) & 0xF9) )
  {
    sub_2EBE740(a1, a2, (8 * v8) | (4LL * v9) | v11 & 1 | (2LL * v10), a4, v12, v17);
    return 1;
  }
  return result;
}
