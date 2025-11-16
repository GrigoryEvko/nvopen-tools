// Function: sub_1B995D0
// Address: 0x1b995d0
//
bool __fastcall sub_1B995D0(_QWORD **a1, int *a2)
{
  int v2; // r12d
  __int64 v4; // r13
  __int64 v5; // r14
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // r9
  unsigned int v11; // ecx
  int *v12; // rdx
  int v13; // r8d
  int v14; // eax
  __int64 v15; // r8
  int v16; // edx
  int v17; // r9d
  unsigned int v18; // eax
  __int64 v19; // rcx
  int v21; // edx
  int v22; // r11d
  int v23; // [rsp+4h] [rbp-2Ch] BYREF
  __int64 v24[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *a2;
  if ( *a2 == 1 )
    return 0;
  v4 = (*a1)[4];
  v5 = *a1[1];
  v23 = *a2;
  v6 = (unsigned __int8)sub_1B97860(v4 + 200, &v23, v24)
     ? v24[0]
     : *(_QWORD *)(v4 + 208) + 80LL * *(unsigned int *)(v4 + 224);
  if ( sub_13A0E30(v6 + 8, v5) )
    return 0;
  v7 = (*a1)[4];
  v8 = *a1[1];
  v9 = *(unsigned int *)(v7 + 160);
  v10 = *(_QWORD *)(v7 + 144);
  if ( (_DWORD)v9 )
  {
    v11 = (v9 - 1) & (37 * v2);
    v12 = (int *)(v10 + 40LL * v11);
    v13 = *v12;
    if ( v2 == *v12 )
      goto LABEL_7;
    v21 = 1;
    while ( v13 != -1 )
    {
      v22 = v21 + 1;
      v11 = (v9 - 1) & (v21 + v11);
      v12 = (int *)(v10 + 40LL * v11);
      v13 = *v12;
      if ( v2 == *v12 )
        goto LABEL_7;
      v21 = v22;
    }
  }
  v12 = (int *)(v10 + 40 * v9);
LABEL_7:
  v14 = v12[8];
  if ( v14 )
  {
    v15 = *((_QWORD *)v12 + 2);
    v16 = v14 - 1;
    v17 = 1;
    v18 = (v14 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v19 = *(_QWORD *)(v15 + 16LL * v18);
    if ( v8 == v19 )
      return 0;
    while ( v19 != -8 )
    {
      v18 = v16 & (v17 + v18);
      v19 = *(_QWORD *)(v15 + 16LL * v18);
      if ( v8 == v19 )
        return 0;
      ++v17;
    }
  }
  return (unsigned int)sub_1B99570(v7, v8, v2) != 5;
}
