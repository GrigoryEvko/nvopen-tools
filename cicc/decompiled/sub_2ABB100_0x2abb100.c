// Function: sub_2ABB100
// Address: 0x2abb100
//
void *__fastcall sub_2ABB100(__int64 a1, char a2, __int64 *a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v8; // r8
  __int64 v9; // r12
  __int64 v10; // rdx
  __int64 *v11; // r13
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v16; // [rsp+8h] [rbp-58h]
  __int64 v17; // [rsp+8h] [rbp-58h]
  __int64 v19[7]; // [rsp+28h] [rbp-38h] BYREF

  v19[0] = *a6;
  if ( v19[0] )
    sub_2AAAFA0(v19);
  *(_BYTE *)(a1 + 8) = a2;
  v8 = (__int64)&a3[a4];
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_4A231A8;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_QWORD *)(a1 + 40) = &unk_4A23170;
  *(_QWORD *)(a1 + 56) = 0x200000000LL;
  if ( (__int64 *)v8 != a3 )
  {
    v9 = *a3;
    v10 = a1 + 64;
    v11 = a3 + 1;
    v12 = 0;
    while ( 1 )
    {
      *(_QWORD *)(v10 + 8 * v12) = v9;
      ++*(_DWORD *)(a1 + 56);
      v13 = *(unsigned int *)(v9 + 24);
      if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(v9 + 28) )
      {
        v17 = v8;
        sub_C8D5F0(v9 + 16, (const void *)(v9 + 32), v13 + 1, 8u, v8, (__int64)a6);
        v13 = *(unsigned int *)(v9 + 24);
        v8 = v17;
      }
      *(_QWORD *)(*(_QWORD *)(v9 + 16) + 8 * v13) = a1 + 40;
      ++*(_DWORD *)(v9 + 24);
      if ( (__int64 *)v8 == v11 )
        break;
      v12 = *(unsigned int *)(a1 + 56);
      v9 = *v11;
      if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 60) )
      {
        v16 = v8;
        sub_C8D5F0(a1 + 48, (const void *)(a1 + 64), v12 + 1, 8u, v8, (__int64)a6);
        v12 = *(unsigned int *)(a1 + 56);
        v8 = v16;
      }
      v10 = *(_QWORD *)(a1 + 48);
      ++v11;
    }
  }
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 40) = &unk_4A23AA8;
  v14 = v19[0];
  *(_QWORD *)a1 = &unk_4A23A70;
  *(_QWORD *)(a1 + 88) = v14;
  if ( v14 )
  {
    sub_2AAAFA0((__int64 *)(a1 + 88));
    if ( v19[0] )
      sub_B91220((__int64)v19, v19[0]);
  }
  sub_2BF0340(a1 + 96, 1, a5, a1);
  *(_QWORD *)a1 = &unk_4A231C8;
  *(_QWORD *)(a1 + 40) = &unk_4A23200;
  *(_QWORD *)(a1 + 96) = &unk_4A23238;
  return &unk_4A23238;
}
