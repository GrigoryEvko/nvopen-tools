// Function: sub_2D58400
// Address: 0x2d58400
//
__int64 __fastcall sub_2D58400(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // rbx
  _QWORD *v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  _QWORD *v8; // r14
  __int64 v9; // r15
  __int64 v10; // r12
  __int64 v11; // r8
  unsigned __int64 v12; // r9
  __int64 v13; // rax
  _QWORD *v14; // rax
  char *v15; // rbx
  __int64 v16; // rdx
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // rsi
  int v19; // eax
  _QWORD *v20; // rdx
  __int64 result; // rax
  char *v22; // rbx
  unsigned __int64 v23; // [rsp+0h] [rbp-70h]
  __int64 i; // [rsp+8h] [rbp-68h]
  const void *v25; // [rsp+10h] [rbp-60h]
  _QWORD v27[7]; // [rsp+38h] [rbp-38h] BYREF

  v5 = (_QWORD *)sub_22077B0(0x98u);
  v8 = v5;
  if ( v5 )
  {
    v5[1] = a2;
    v9 = *(_QWORD *)(a2 + 16);
    *v5 = off_49D4150;
    v25 = v5 + 4;
    v5[2] = v5 + 4;
    v5[3] = 0x400000000LL;
    v5[12] = v5 + 14;
    v5[13] = 0x100000000LL;
    v5[16] = 0x100000000LL;
    v5[18] = a3;
    v5[15] = v5 + 17;
    for ( i = (__int64)(v5 + 2); v9; v9 = *(_QWORD *)(v9 + 8) )
    {
      v10 = *(_QWORD *)(v9 + 24);
      v12 = (unsigned int)sub_BD2910(v9) | v3 & 0xFFFFFFFF00000000LL;
      v13 = *((unsigned int *)v8 + 6);
      v3 = v12;
      if ( v13 + 1 > (unsigned __int64)*((unsigned int *)v8 + 7) )
      {
        v23 = v12;
        sub_C8D5F0(i, v25, v13 + 1, 0x10u, v11, v12);
        v13 = *((unsigned int *)v8 + 6);
        v12 = v23;
      }
      v14 = (_QWORD *)(v8[2] + 16 * v13);
      *v14 = v10;
      v14[1] = v12;
      ++*((_DWORD *)v8 + 6);
    }
    sub_AE7A40((__int64)(v8 + 12), (_BYTE *)a2, (__int64)(v8 + 15));
    sub_BD84D0(a2, a3);
  }
  v27[0] = v8;
  v15 = (char *)v27;
  v16 = *(unsigned int *)(a1 + 8);
  v17 = *(_QWORD *)a1;
  v18 = v16 + 1;
  v19 = *(_DWORD *)(a1 + 8);
  if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    if ( v17 > (unsigned __int64)v27 || (unsigned __int64)v27 >= v17 + 8 * v16 )
    {
      sub_2D57B00(a1, v18, v16, v17, v6, v7);
      v16 = *(unsigned int *)(a1 + 8);
      v17 = *(_QWORD *)a1;
      v19 = *(_DWORD *)(a1 + 8);
    }
    else
    {
      v22 = (char *)v27 - v17;
      sub_2D57B00(a1, v18, v16, v17, v6, v7);
      v17 = *(_QWORD *)a1;
      v16 = *(unsigned int *)(a1 + 8);
      v15 = &v22[*(_QWORD *)a1];
      v19 = *(_DWORD *)(a1 + 8);
    }
  }
  v20 = (_QWORD *)(v17 + 8 * v16);
  if ( v20 )
  {
    *v20 = *(_QWORD *)v15;
    *(_QWORD *)v15 = 0;
    v8 = (_QWORD *)v27[0];
    v19 = *(_DWORD *)(a1 + 8);
  }
  result = (unsigned int)(v19 + 1);
  *(_DWORD *)(a1 + 8) = result;
  if ( v8 )
    return (*(__int64 (__fastcall **)(_QWORD *))(*v8 + 8LL))(v8);
  return result;
}
