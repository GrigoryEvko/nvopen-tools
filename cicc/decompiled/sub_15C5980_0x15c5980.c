// Function: sub_15C5980
// Address: 0x15c5980
//
unsigned int *__fastcall sub_15C5980(unsigned int *a1, __int64 a2, __int64 a3)
{
  unsigned int *result; // rax
  unsigned int v4; // r12d
  int v6; // esi
  unsigned int **v7; // rcx
  unsigned int *v8; // rdi
  int v9; // eax
  __int64 v10; // rax
  __int64 v11; // r13
  unsigned int v12; // r12d
  unsigned int v13; // edx
  unsigned int **v14; // rsi
  int v15; // r8d
  unsigned int **v16; // r9
  int v17; // eax
  unsigned int *v18; // [rsp+8h] [rbp-68h] BYREF
  unsigned int **v19; // [rsp+10h] [rbp-60h] BYREF
  __int64 v20; // [rsp+18h] [rbp-58h] BYREF
  unsigned int v21; // [rsp+20h] [rbp-50h] BYREF
  __int64 v22; // [rsp+28h] [rbp-48h] BYREF
  __int64 v23; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v24; // [rsp+38h] [rbp-38h] BYREF
  __int64 v25[6]; // [rsp+40h] [rbp-30h] BYREF

  v18 = a1;
  if ( (_DWORD)a2 )
  {
    result = a1;
    if ( (_DWORD)a2 == 1 )
    {
      sub_1621390(a1, a2);
      return v18;
    }
    return result;
  }
  v4 = *(_DWORD *)(a3 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a3;
LABEL_7:
    v6 = 2 * v4;
    goto LABEL_8;
  }
  v10 = a1[2];
  v11 = *(_QWORD *)(a3 + 8);
  v12 = v4 - 1;
  v19 = *(unsigned int ***)&a1[-2 * v10];
  v20 = *(_QWORD *)&a1[2 * (1 - v10)];
  v21 = a1[6];
  v22 = *(_QWORD *)&a1[2 * (2 - v10)];
  v23 = *(_QWORD *)&a1[2 * (3 - v10)];
  v24 = a1[7];
  v25[0] = *(_QWORD *)&a1[2 * (4 - v10)];
  v8 = v18;
  v13 = v12 & sub_15B52D0((__int64 *)&v19, &v20, (int *)&v21, &v22, &v23, (int *)&v24, v25);
  v14 = (unsigned int **)(v11 + 8LL * v13);
  result = *v14;
  if ( v18 == *v14 )
    return result;
  v15 = 1;
  v7 = 0;
  while ( result != (unsigned int *)-8LL )
  {
    if ( result != (unsigned int *)-16LL || v7 )
      v14 = v7;
    v13 = v12 & (v15 + v13);
    v16 = (unsigned int **)(v11 + 8LL * v13);
    result = *v16;
    if ( *v16 == v18 )
      return result;
    ++v15;
    v7 = v14;
    v14 = (unsigned int **)(v11 + 8LL * v13);
  }
  v17 = *(_DWORD *)(a3 + 16);
  v4 = *(_DWORD *)(a3 + 24);
  if ( !v7 )
    v7 = v14;
  ++*(_QWORD *)a3;
  v9 = v17 + 1;
  if ( 4 * v9 >= 3 * v4 )
    goto LABEL_7;
  if ( v4 - (v9 + *(_DWORD *)(a3 + 20)) > v4 >> 3 )
    goto LABEL_9;
  v6 = v4;
LABEL_8:
  sub_15C5700(a3, v6);
  sub_15B93D0(a3, &v18, &v19);
  v7 = v19;
  v8 = v18;
  v9 = *(_DWORD *)(a3 + 16) + 1;
LABEL_9:
  *(_DWORD *)(a3 + 16) = v9;
  if ( *v7 != (unsigned int *)-8LL )
    --*(_DWORD *)(a3 + 20);
  *v7 = v8;
  return v18;
}
