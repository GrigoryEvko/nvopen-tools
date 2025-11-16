// Function: sub_1484870
// Address: 0x1484870
//
__int64 __fastcall sub_1484870(_QWORD *a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5)
{
  __int64 v7; // rdi
  unsigned int v8; // ebx
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // r15
  unsigned int v12; // r12d
  unsigned __int64 v13; // rax
  unsigned int v14; // r12d
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax

  if ( *(_WORD *)(a3 + 24) )
    goto LABEL_11;
  v7 = *(_QWORD *)(a3 + 32);
  v8 = *(_DWORD *)(v7 + 32);
  if ( v8 > 0x40 )
  {
    if ( (unsigned int)sub_16A57B0(v7 + 24) != v8 - 1 )
    {
      if ( (unsigned int)sub_16A5940(v7 + 24) == 1 )
        goto LABEL_5;
LABEL_11:
      v20 = sub_1483CF0(a1, a2, a3, a4, a5);
      v21 = sub_13A5B60((__int64)a1, v20, a3, 2u, 0);
      return sub_14806B0((__int64)a1, a2, v21, 2, 0);
    }
LABEL_12:
    v22 = sub_1456040(a2);
    return sub_145CF80((__int64)a1, v22, 0, 0);
  }
  v19 = *(_QWORD *)(v7 + 24);
  if ( v19 == 1 )
    goto LABEL_12;
  if ( !v19 || (v19 & (v19 - 1)) != 0 )
    goto LABEL_11;
LABEL_5:
  v9 = sub_1456040(a2);
  v10 = *(_QWORD *)(a3 + 32);
  v11 = v9;
  v12 = *(_DWORD *)(v10 + 32);
  if ( v12 > 0x40 )
  {
    v14 = v12 - 1 - sub_16A57B0(v10 + 24);
  }
  else
  {
    v13 = *(_QWORD *)(v10 + 24);
    v14 = -1;
    if ( v13 )
    {
      _BitScanReverse64(&v13, v13);
      v14 = 63 - (v13 ^ 0x3F);
    }
  }
  v15 = sub_15E0530(a1[3]);
  v16 = sub_1644900(v15, v14);
  v17 = sub_14835F0(a1, a2, v16, 0, a4, a5);
  return sub_14747F0((__int64)a1, v17, v11, 0);
}
