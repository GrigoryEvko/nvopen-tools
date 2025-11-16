// Function: sub_22B6470
// Address: 0x22b6470
//
__int64 __fastcall sub_22B6470(__int64 a1, __int64 a2, int *a3)
{
  __int64 v6; // rsi
  __int64 v7; // r8
  int v8; // eax
  __int64 v9; // r9
  int v10; // r14d
  int *v11; // r11
  unsigned int v12; // ecx
  int *v13; // rdx
  int v14; // edi
  __int64 v15; // rsi
  char v16; // cl
  int v18; // eax
  int v19; // ecx
  int *v20; // [rsp+8h] [rbp-38h] BYREF

  v6 = *(unsigned int *)(a2 + 24);
  v7 = *(_QWORD *)a2;
  if ( !(_DWORD)v6 )
  {
    v20 = 0;
    *(_QWORD *)a2 = v7 + 1;
LABEL_19:
    LODWORD(v6) = 2 * v6;
    goto LABEL_20;
  }
  v8 = *a3;
  v9 = *(_QWORD *)(a2 + 8);
  v10 = 1;
  v11 = 0;
  v12 = (v6 - 1) & (37 * *a3);
  v13 = (int *)(v9 + 4LL * v12);
  v14 = *v13;
  if ( v8 == *v13 )
  {
LABEL_3:
    v15 = v9 + 4 * v6;
    v16 = 0;
    goto LABEL_4;
  }
  while ( v14 != -1 )
  {
    if ( v14 == -2 && !v11 )
      v11 = v13;
    v12 = (v6 - 1) & (v10 + v12);
    v13 = (int *)(v9 + 4LL * v12);
    v14 = *v13;
    if ( v8 == *v13 )
      goto LABEL_3;
    ++v10;
  }
  *(_QWORD *)a2 = v7 + 1;
  v18 = *(_DWORD *)(a2 + 16);
  if ( v11 )
    v13 = v11;
  v19 = v18 + 1;
  v20 = v13;
  if ( 4 * (v18 + 1) >= (unsigned int)(3 * v6) )
    goto LABEL_19;
  if ( (int)v6 - *(_DWORD *)(a2 + 20) - v19 <= (unsigned int)v6 >> 3 )
  {
LABEL_20:
    sub_A08C50(a2, v6);
    sub_22B31A0(a2, a3, &v20);
    v13 = v20;
    v19 = *(_DWORD *)(a2 + 16) + 1;
  }
  *(_DWORD *)(a2 + 16) = v19;
  if ( *v13 != -1 )
    --*(_DWORD *)(a2 + 20);
  *v13 = *a3;
  v7 = *(_QWORD *)a2;
  v15 = *(_QWORD *)(a2 + 8) + 4LL * *(unsigned int *)(a2 + 24);
  v16 = 1;
LABEL_4:
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = v7;
  *(_QWORD *)(a1 + 16) = v13;
  *(_QWORD *)(a1 + 24) = v15;
  *(_BYTE *)(a1 + 32) = v16;
  return a1;
}
