// Function: sub_F86C90
// Address: 0xf86c90
//
__int64 __fastcall sub_F86C90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rsi
  __int64 v7; // rdx
  _QWORD *v8; // r13
  int v9; // edx
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rcx
  char v13; // al
  __int64 v14; // rcx
  __int64 v15; // rdi
  unsigned int v16; // eax
  __int64 v17; // r8
  int v19; // r11d
  _QWORD *v20; // r10
  int v21; // eax
  _QWORD *v22; // [rsp+8h] [rbp-38h] BYREF

  v6 = *(unsigned int *)(a2 + 24);
  v7 = *(_QWORD *)a2;
  if ( !(_DWORD)v6 )
  {
    v22 = 0;
    *(_QWORD *)a2 = v7 + 1;
LABEL_3:
    LODWORD(v6) = 2 * v6;
LABEL_4:
    sub_F86930(a2, v6);
    sub_F82F60(a2, a3, &v22);
    v8 = v22;
    v9 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_5;
  }
  v14 = *(_QWORD *)(a3 + 16);
  v15 = *(_QWORD *)(a2 + 8);
  v16 = (v6 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
  v8 = (_QWORD *)(v15 + 24LL * v16);
  v17 = v8[2];
  if ( v17 == v14 )
  {
LABEL_15:
    v12 = v15 + 24 * v6;
    v13 = 0;
    goto LABEL_16;
  }
  v19 = 1;
  v20 = 0;
  while ( v17 != -4096 )
  {
    if ( v17 == -8192 && !v20 )
      v20 = v8;
    v16 = (v6 - 1) & (v19 + v16);
    v8 = (_QWORD *)(v15 + 24LL * v16);
    v17 = v8[2];
    if ( v14 == v17 )
      goto LABEL_15;
    ++v19;
  }
  v21 = *(_DWORD *)(a2 + 16);
  if ( v20 )
    v8 = v20;
  *(_QWORD *)a2 = v7 + 1;
  v9 = v21 + 1;
  v22 = v8;
  if ( 4 * (v21 + 1) >= (unsigned int)(3 * v6) )
    goto LABEL_3;
  if ( (int)v6 - *(_DWORD *)(a2 + 20) - v9 <= (unsigned int)v6 >> 3 )
    goto LABEL_4;
LABEL_5:
  *(_DWORD *)(a2 + 16) = v9;
  if ( v8[2] == -4096 )
  {
    v11 = *(_QWORD *)(a3 + 16);
    if ( v11 != -4096 )
    {
LABEL_10:
      v8[2] = v11;
      if ( v11 != 0 && v11 != -4096 && v11 != -8192 )
        sub_BD73F0((__int64)v8);
    }
  }
  else
  {
    --*(_DWORD *)(a2 + 20);
    v10 = v8[2];
    v11 = *(_QWORD *)(a3 + 16);
    if ( v11 != v10 )
    {
      if ( v10 != 0 && v10 != -4096 && v10 != -8192 )
        sub_BD60C0(v8);
      goto LABEL_10;
    }
  }
  v12 = *(_QWORD *)(a2 + 8) + 24LL * *(unsigned int *)(a2 + 24);
  v7 = *(_QWORD *)a2;
  v13 = 1;
LABEL_16:
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v8;
  *(_BYTE *)(a1 + 32) = v13;
  *(_QWORD *)(a1 + 8) = v7;
  *(_QWORD *)(a1 + 24) = v12;
  return a1;
}
