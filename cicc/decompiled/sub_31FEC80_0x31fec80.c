// Function: sub_31FEC80
// Address: 0x31fec80
//
__int64 __fastcall sub_31FEC80(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v4; // r13
  unsigned int v7; // r8d
  __int64 v8; // rdx
  int v9; // r14d
  __int64 *v10; // rdi
  unsigned int i; // eax
  __int64 *v12; // r9
  __int64 v13; // r11
  unsigned int v14; // eax
  int v16; // eax
  int v17; // edx
  int v18; // esi
  __int64 *v19; // [rsp+8h] [rbp-48h] BYREF
  __int64 v20; // [rsp+10h] [rbp-40h] BYREF
  __int64 v21; // [rsp+18h] [rbp-38h]
  unsigned int v22; // [rsp+20h] [rbp-30h]

  v4 = a1 + 1216;
  v7 = *(_DWORD *)(a1 + 1240);
  v20 = a2;
  v21 = a4;
  v22 = a3;
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 1216);
    v19 = 0;
    goto LABEL_23;
  }
  v8 = *(_QWORD *)(a1 + 1224);
  v9 = 1;
  v10 = 0;
  for ( i = (v7 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4)
              | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4)))); ; i = (v7 - 1) & v14 )
  {
    v12 = (__int64 *)(v8 + 24LL * i);
    v13 = *v12;
    if ( a2 == *v12 && a4 == v12[1] )
      return a3;
    if ( v13 == -4096 )
      break;
    if ( v13 == -8192 && v12[1] == -8192 && !v10 )
      v10 = (__int64 *)(v8 + 24LL * i);
LABEL_9:
    v14 = v9 + i;
    ++v9;
  }
  if ( v12[1] != -4096 )
    goto LABEL_9;
  v16 = *(_DWORD *)(a1 + 1232);
  if ( !v10 )
    v10 = v12;
  ++*(_QWORD *)(a1 + 1216);
  v17 = v16 + 1;
  v19 = v10;
  if ( 4 * (v16 + 1) >= 3 * v7 )
  {
LABEL_23:
    v18 = 2 * v7;
    goto LABEL_24;
  }
  if ( v7 - *(_DWORD *)(a1 + 1236) - v17 > v7 >> 3 )
    goto LABEL_17;
  v18 = v7;
LABEL_24:
  sub_31FE9B0(v4, v18);
  sub_31FB320(v4, &v20, &v19);
  a2 = v20;
  v10 = v19;
  v17 = *(_DWORD *)(a1 + 1232) + 1;
LABEL_17:
  *(_DWORD *)(a1 + 1232) = v17;
  if ( *v10 != -4096 || v10[1] != -4096 )
    --*(_DWORD *)(a1 + 1236);
  *v10 = a2;
  v10[1] = v21;
  *((_DWORD *)v10 + 4) = v22;
  return a3;
}
