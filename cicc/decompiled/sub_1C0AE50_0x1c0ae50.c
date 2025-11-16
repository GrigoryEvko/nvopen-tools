// Function: sub_1C0AE50
// Address: 0x1c0ae50
//
__int64 __fastcall sub_1C0AE50(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v6; // rsi
  __int64 v7; // r8
  __int64 v8; // rcx
  __int64 v9; // r9
  __int64 *v10; // r11
  int v11; // r14d
  unsigned int v12; // eax
  __int64 *v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rsi
  char v16; // cl
  int v18; // eax
  int v19; // ecx
  __int64 *v20; // [rsp+8h] [rbp-38h] BYREF

  v6 = *(unsigned int *)(a2 + 24);
  v7 = *(_QWORD *)a2;
  if ( !(_DWORD)v6 )
  {
    *(_QWORD *)a2 = v7 + 1;
LABEL_19:
    LODWORD(v6) = 2 * v6;
    goto LABEL_20;
  }
  v8 = *a3;
  v9 = *(_QWORD *)(a2 + 8);
  v10 = 0;
  v11 = 1;
  v12 = (v6 - 1) & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
  v13 = (__int64 *)(v9 + 8LL * v12);
  v14 = *v13;
  if ( v8 == *v13 )
  {
LABEL_3:
    v15 = v9 + 8 * v6;
    v16 = 0;
    goto LABEL_4;
  }
  while ( v14 != -8 )
  {
    if ( v14 == -16 && !v10 )
      v10 = v13;
    v12 = (v6 - 1) & (v11 + v12);
    v13 = (__int64 *)(v9 + 8LL * v12);
    v14 = *v13;
    if ( v8 == *v13 )
      goto LABEL_3;
    ++v11;
  }
  *(_QWORD *)a2 = v7 + 1;
  v18 = *(_DWORD *)(a2 + 16);
  if ( v10 )
    v13 = v10;
  v19 = v18 + 1;
  if ( 4 * (v18 + 1) >= (unsigned int)(3 * v6) )
    goto LABEL_19;
  if ( (int)v6 - *(_DWORD *)(a2 + 20) - v19 <= (unsigned int)v6 >> 3 )
  {
LABEL_20:
    sub_1353F00(a2, v6);
    sub_1A97120(a2, a3, &v20);
    v13 = v20;
    v19 = *(_DWORD *)(a2 + 16) + 1;
  }
  *(_DWORD *)(a2 + 16) = v19;
  if ( *v13 != -8 )
    --*(_DWORD *)(a2 + 20);
  *v13 = *a3;
  v7 = *(_QWORD *)a2;
  v15 = *(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24);
  v16 = 1;
LABEL_4:
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = v7;
  *(_QWORD *)(a1 + 16) = v13;
  *(_QWORD *)(a1 + 24) = v15;
  *(_BYTE *)(a1 + 32) = v16;
  return a1;
}
