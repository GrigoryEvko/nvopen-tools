// Function: sub_239C850
// Address: 0x239c850
//
__int64 __fastcall sub_239C850(__int64 a1, __int64 a2)
{
  __int64 *v3; // rcx
  __int64 v4; // r8
  __int64 v5; // rsi
  __int64 v6; // r12
  char v7; // r9
  __int64 v8; // rdi
  int v9; // r10d
  unsigned int v10; // edx
  __int64 v11; // rax
  __int64 v12; // r11
  __int64 v13; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdi
  int v19; // r13d
  unsigned int i; // eax
  _QWORD *v21; // r9
  unsigned int v22; // eax
  __int64 v23; // rax
  int v24; // eax
  char v25; // al
  int v26; // r13d
  __int64 v27; // [rsp+0h] [rbp-60h] BYREF
  char v28[8]; // [rsp+8h] [rbp-58h] BYREF
  char v29[16]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v30; // [rsp+20h] [rbp-40h]

  v3 = *(__int64 **)a1;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = **(_QWORD **)a1;
  v7 = *(_BYTE *)(v6 + 8) & 1;
  if ( v7 )
  {
    v8 = v6 + 16;
    v9 = 7;
  }
  else
  {
    v15 = *(unsigned int *)(v6 + 24);
    v8 = *(_QWORD *)(v6 + 16);
    if ( !(_DWORD)v15 )
      goto LABEL_16;
    v9 = v15 - 1;
  }
  v10 = v9 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = v8 + 16LL * v10;
  v12 = *(_QWORD *)v11;
  if ( a2 == *(_QWORD *)v11 )
    goto LABEL_4;
  v24 = 1;
  while ( v12 != -4096 )
  {
    v26 = v24 + 1;
    v10 = v9 & (v24 + v10);
    v11 = v8 + 16LL * v10;
    v12 = *(_QWORD *)v11;
    if ( a2 == *(_QWORD *)v11 )
      goto LABEL_4;
    v24 = v26;
  }
  if ( v7 )
  {
    v23 = 128;
    goto LABEL_17;
  }
  v15 = *(unsigned int *)(v6 + 24);
LABEL_16:
  v23 = 16 * v15;
LABEL_17:
  v11 = v8 + v23;
LABEL_4:
  v13 = 128;
  if ( !v7 )
    v13 = 16LL * *(unsigned int *)(v6 + 24);
  if ( v11 != v8 + v13 )
    return *(unsigned __int8 *)(v11 + 8);
  v16 = v3[1];
  v17 = *(unsigned int *)(v16 + 24);
  v18 = *(_QWORD *)(v16 + 8);
  if ( (_DWORD)v17 )
  {
    v19 = 1;
    for ( i = (v17 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)
                | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)))); ; i = (v17 - 1) & v22 )
    {
      v21 = (_QWORD *)(v18 + 24LL * i);
      if ( a2 == *v21 && v5 == v21[1] )
        break;
      if ( *v21 == -4096 && v21[1] == -4096 )
        goto LABEL_23;
      v22 = v19 + i;
      ++v19;
    }
  }
  else
  {
LABEL_23:
    v21 = (_QWORD *)(v18 + 24 * v17);
  }
  v25 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(v21[2] + 24LL) + 16LL))(
          *(_QWORD *)(v21[2] + 24LL),
          v5,
          v4);
  v27 = a2;
  v28[0] = v25;
  sub_BBCF50((__int64)v29, v6, &v27, v28);
  return *(unsigned __int8 *)(v30 + 8);
}
