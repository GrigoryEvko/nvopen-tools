// Function: sub_22D06C0
// Address: 0x22d06c0
//
__int64 __fastcall sub_22D06C0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r12
  char v8; // r9
  __int64 v9; // rcx
  int v10; // r10d
  unsigned int v11; // edx
  __int64 v12; // rax
  __int64 v13; // r11
  __int64 v14; // rdx
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  int v20; // r13d
  unsigned int i; // eax
  _QWORD *v22; // r9
  unsigned int v23; // eax
  __int64 v24; // rax
  int v25; // eax
  char v26; // al
  int v27; // r13d
  __int64 v28; // [rsp+0h] [rbp-60h] BYREF
  char v29[8]; // [rsp+8h] [rbp-58h] BYREF
  char v30[16]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v31; // [rsp+20h] [rbp-40h]

  v7 = *a1;
  v8 = *(_BYTE *)(*a1 + 8) & 1;
  if ( v8 )
  {
    v9 = v7 + 16;
    v10 = 7;
  }
  else
  {
    v16 = *(unsigned int *)(v7 + 24);
    v9 = *(_QWORD *)(v7 + 16);
    if ( !(_DWORD)v16 )
      goto LABEL_16;
    v10 = v16 - 1;
  }
  v11 = v10 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v12 = v9 + 16LL * v11;
  v13 = *(_QWORD *)v12;
  if ( a2 == *(_QWORD *)v12 )
    goto LABEL_4;
  v25 = 1;
  while ( v13 != -4096 )
  {
    v27 = v25 + 1;
    v11 = v10 & (v25 + v11);
    v12 = v9 + 16LL * v11;
    v13 = *(_QWORD *)v12;
    if ( a2 == *(_QWORD *)v12 )
      goto LABEL_4;
    v25 = v27;
  }
  if ( v8 )
  {
    v24 = 128;
    goto LABEL_17;
  }
  v16 = *(unsigned int *)(v7 + 24);
LABEL_16:
  v24 = 16 * v16;
LABEL_17:
  v12 = v9 + v24;
LABEL_4:
  v14 = 128;
  if ( !v8 )
    v14 = 16LL * *(unsigned int *)(v7 + 24);
  if ( v12 != v9 + v14 )
    return *(unsigned __int8 *)(v12 + 8);
  v17 = a1[1];
  v18 = *(unsigned int *)(v17 + 24);
  v19 = *(_QWORD *)(v17 + 8);
  if ( (_DWORD)v18 )
  {
    v20 = 1;
    for ( i = (v18 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v18 - 1) & v23 )
    {
      v22 = (_QWORD *)(v19 + 24LL * i);
      if ( a2 == *v22 && a3 == v22[1] )
        break;
      if ( *v22 == -4096 && v22[1] == -4096 )
        goto LABEL_23;
      v23 = v20 + i;
      ++v20;
    }
  }
  else
  {
LABEL_23:
    v22 = (_QWORD *)(v19 + 24 * v18);
  }
  v26 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64 *))(**(_QWORD **)(v22[2] + 24LL) + 16LL))(
          *(_QWORD *)(v22[2] + 24LL),
          a3,
          a4,
          a1);
  v28 = a2;
  v29[0] = v26;
  sub_BBCF50((__int64)v30, v7, &v28, v29);
  return *(unsigned __int8 *)(v31 + 8);
}
