// Function: sub_2B153A0
// Address: 0x2b153a0
//
__int64 __fastcall sub_2B153A0(__int64 **a1)
{
  __int64 v1; // rcx
  unsigned __int64 v2; // rax
  _QWORD *v3; // r14
  int v4; // r15d
  _QWORD *v5; // r12
  __int64 v6; // r13
  int v7; // r15d
  __int64 v8; // r13
  __int64 v9; // r12
  int v10; // r15d
  __int64 v11; // r13
  int v12; // r15d
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // r12
  unsigned __int64 v18; // [rsp+8h] [rbp-B8h]
  unsigned __int64 v19; // [rsp+10h] [rbp-B0h]
  __int64 v20; // [rsp+18h] [rbp-A8h]
  __int64 v21; // [rsp+28h] [rbp-98h]
  unsigned __int64 v22; // [rsp+30h] [rbp-90h]
  __int64 v23; // [rsp+38h] [rbp-88h]
  _QWORD *v24; // [rsp+40h] [rbp-80h]
  __int64 v25; // [rsp+48h] [rbp-78h]
  char *v26; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v27; // [rsp+58h] [rbp-68h]
  char v28; // [rsp+60h] [rbp-60h] BYREF

  v1 = **a1;
  v20 = (*a1)[1];
  v2 = *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)v1 - 64LL) + 8LL) + 32LL)
     / (unsigned __int64)*(unsigned int *)(*(_QWORD *)v1 + 80LL);
  v19 = v2;
  if ( !v20 )
    return 0;
  v18 = *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)**a1 - 64LL) + 8LL) + 32LL)
      / (unsigned __int64)*(unsigned int *)(*(_QWORD *)**a1 + 80LL);
  v22 = v2 >> 2;
  v23 = v2;
  v21 = 4 * (v2 >> 2);
  v25 = 0;
  while ( 1 )
  {
    v3 = (_QWORD *)(v1 + 8 * v25);
    v24 = &v3[v23];
    if ( v22 )
      break;
    v14 = v18;
    v4 = 0;
LABEL_27:
    if ( v14 != 2 )
    {
      if ( v14 != 3 )
      {
        if ( v14 != 1 )
          goto LABEL_4;
        goto LABEL_30;
      }
      v15 = *v3;
      if ( *(_BYTE *)(*(_QWORD *)(*v3 + 8LL) + 8LL) != 18 )
        sub_B4EFF0(
          *(int **)(v15 + 72),
          *(unsigned int *)(v15 + 80),
          *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v15 - 64) + 8LL) + 32LL),
          (int *)&v26);
      if ( (_DWORD)v26 != v4 )
        goto LABEL_3;
      v4 += *(_DWORD *)(v15 + 80);
      ++v3;
    }
    v16 = *v3;
    if ( *(_BYTE *)(*(_QWORD *)(*v3 + 8LL) + 8LL) != 18 )
      sub_B4EFF0(
        *(int **)(v16 + 72),
        *(unsigned int *)(v16 + 80),
        *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v16 - 64) + 8LL) + 32LL),
        (int *)&v26);
    if ( (_DWORD)v26 != v4 )
      goto LABEL_3;
    v4 += *(_DWORD *)(v16 + 80);
    ++v3;
LABEL_30:
    if ( *(_BYTE *)(*(_QWORD *)(*v3 + 8LL) + 8LL) != 18 )
      sub_B4EFF0(
        *(int **)(*v3 + 72LL),
        *(unsigned int *)(*v3 + 80LL),
        *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(*v3 - 64LL) + 8LL) + 32LL),
        (int *)&v26);
    if ( v4 != (_DWORD)v26 )
      goto LABEL_3;
LABEL_4:
    v25 += v19;
    if ( v20 == v25 )
      return 0;
    v1 = **a1;
  }
  v4 = 0;
  v5 = &v3[v21];
  while ( 1 )
  {
    v6 = *v3;
    if ( *(_BYTE *)(*(_QWORD *)(*v3 + 8LL) + 8LL) != 18 )
      sub_B4EFF0(
        *(int **)(v6 + 72),
        *(unsigned int *)(v6 + 80),
        *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v6 - 64) + 8LL) + 32LL),
        (int *)&v26);
    if ( (_DWORD)v26 != v4 )
      break;
    v7 = *(_DWORD *)(v6 + 80) + v4;
    v8 = v3[1];
    if ( *(_BYTE *)(*(_QWORD *)(v8 + 8) + 8LL) != 18 )
      sub_B4EFF0(
        *(int **)(v8 + 72),
        *(unsigned int *)(v8 + 80),
        *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v8 - 64) + 8LL) + 32LL),
        (int *)&v26);
    if ( v7 != (_DWORD)v26 )
    {
      if ( v24 == v3 + 1 )
        goto LABEL_4;
      goto LABEL_15;
    }
    v10 = *(_DWORD *)(v8 + 80) + v7;
    v11 = v3[2];
    if ( *(_BYTE *)(*(_QWORD *)(v11 + 8) + 8LL) != 18 )
      sub_B4EFF0(
        *(int **)(v11 + 72),
        *(unsigned int *)(v11 + 80),
        *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v11 - 64) + 8LL) + 32LL),
        (int *)&v26);
    if ( v10 != (_DWORD)v26 )
    {
      v3 += 2;
      break;
    }
    v12 = *(_DWORD *)(v11 + 80) + v10;
    v13 = v3[3];
    if ( *(_BYTE *)(*(_QWORD *)(v13 + 8) + 8LL) != 18 )
      sub_B4EFF0(
        *(int **)(v13 + 72),
        *(unsigned int *)(v13 + 80),
        *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v13 - 64) + 8LL) + 32LL),
        (int *)&v26);
    if ( v12 != (_DWORD)v26 )
    {
      v3 += 3;
      break;
    }
    v3 += 4;
    v4 = *(_DWORD *)(v13 + 80) + v12;
    if ( v5 == v3 )
    {
      v14 = v24 - v3;
      goto LABEL_27;
    }
  }
LABEL_3:
  if ( v24 == v3 )
    goto LABEL_4;
LABEL_15:
  sub_2B0F350((__int64)&v26, *(_QWORD **)*a1[3], *(unsigned int *)(*a1[3] + 8));
  v9 = sub_DFBC30((__int64 *)a1[1][412], 7, *a1[2], (__int64)v26, v27, 0, 0, 0, 0, 0, 0);
  if ( v26 != &v28 )
    _libc_free((unsigned __int64)v26);
  return v9;
}
