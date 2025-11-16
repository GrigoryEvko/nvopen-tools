// Function: sub_2AAF310
// Address: 0x2aaf310
//
__int64 __fastcall sub_2AAF310(__int64 a1, char a2, __int64 *a3, __int64 a4, __int64 *a5, __int64 a6)
{
  __int64 v6; // r8
  __int64 v7; // rbx
  __int64 *v8; // r13
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 result; // rax
  __int64 v13; // [rsp+8h] [rbp-48h]
  __int64 v14; // [rsp+8h] [rbp-48h]

  v6 = (__int64)&a3[a4];
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_4A231A8;
  *(_BYTE *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 40) = &unk_4A23170;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_QWORD *)(a1 + 56) = 0x200000000LL;
  if ( (__int64 *)v6 != a3 )
  {
    v7 = *a3;
    v8 = a3 + 1;
    v9 = 0;
    v10 = a1 + 64;
    while ( 1 )
    {
      *(_QWORD *)(v10 + 8 * v9) = v7;
      ++*(_DWORD *)(a1 + 56);
      v11 = *(unsigned int *)(v7 + 24);
      if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(v7 + 28) )
      {
        v14 = v6;
        sub_C8D5F0(v7 + 16, (const void *)(v7 + 32), v11 + 1, 8u, v6, a6);
        v11 = *(unsigned int *)(v7 + 24);
        v6 = v14;
      }
      *(_QWORD *)(*(_QWORD *)(v7 + 16) + 8 * v11) = a1 + 40;
      ++*(_DWORD *)(v7 + 24);
      if ( (__int64 *)v6 == v8 )
        break;
      v9 = *(unsigned int *)(a1 + 56);
      v7 = *v8;
      if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 60) )
      {
        v13 = v6;
        sub_C8D5F0(a1 + 48, (const void *)(a1 + 64), v9 + 1, 8u, v6, a6);
        v9 = *(unsigned int *)(a1 + 56);
        v6 = v13;
      }
      v10 = *(_QWORD *)(a1 + 48);
      ++v8;
    }
  }
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 40) = &unk_4A23AA8;
  *(_QWORD *)a1 = &unk_4A23A70;
  result = *a5;
  *(_QWORD *)(a1 + 88) = *a5;
  if ( result )
    return sub_2AAAFA0((__int64 *)(a1 + 88));
  return result;
}
