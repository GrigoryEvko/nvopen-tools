// Function: sub_2AAFC70
// Address: 0x2aafc70
//
__int64 __fastcall sub_2AAFC70(__int64 a1, char a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 *a7)
{
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned __int64 v14; // rcx
  __int64 v15; // rax
  __int64 result; // rax
  unsigned __int64 v17; // rcx
  __int64 v20; // [rsp+38h] [rbp-48h] BYREF
  __int64 v21; // [rsp+40h] [rbp-40h] BYREF
  __int64 v22[7]; // [rsp+48h] [rbp-38h] BYREF

  v20 = *a7;
  if ( !v20 )
  {
    v21 = 0;
    goto LABEL_15;
  }
  sub_2AAAFA0(&v20);
  v21 = v20;
  if ( !v20 )
  {
LABEL_15:
    v22[0] = 0;
    goto LABEL_5;
  }
  sub_2AAAFA0(&v21);
  v22[0] = v21;
  if ( v21 )
    sub_2AAAFA0(v22);
LABEL_5:
  *(_BYTE *)(a1 + 8) = a2;
  v9 = a1 + 40;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 64) = a4;
  *(_QWORD *)a1 = &unk_4A231A8;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 40) = &unk_4A23170;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_QWORD *)(a1 + 56) = 0x200000001LL;
  v10 = *(unsigned int *)(a4 + 24);
  if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 28) )
  {
    sub_C8D5F0(a4 + 16, (const void *)(a4 + 32), v10 + 1, 8u, v10 + 1, a6);
    v10 = *(unsigned int *)(a4 + 24);
  }
  *(_QWORD *)(*(_QWORD *)(a4 + 16) + 8 * v10) = v9;
  ++*(_DWORD *)(a4 + 24);
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 40) = &unk_4A23AA8;
  v11 = v22[0];
  *(_QWORD *)a1 = &unk_4A23A70;
  *(_QWORD *)(a1 + 88) = v11;
  if ( v11 )
    sub_2AAAFA0((__int64 *)(a1 + 88));
  sub_9C6650(v22);
  sub_2BF0340(a1 + 96, 1, a3, a1);
  *(_QWORD *)a1 = &unk_4A231C8;
  *(_QWORD *)(a1 + 40) = &unk_4A23200;
  *(_QWORD *)(a1 + 96) = &unk_4A23238;
  sub_9C6650(&v21);
  *(_QWORD *)a1 = &unk_4A23FE8;
  *(_QWORD *)(a1 + 40) = &unk_4A24030;
  *(_QWORD *)(a1 + 96) = &unk_4A24068;
  sub_9C6650(&v20);
  v14 = *(unsigned int *)(a1 + 60);
  *(_QWORD *)a1 = &unk_4A232E8;
  *(_QWORD *)(a1 + 96) = &unk_4A23370;
  *(_QWORD *)(a1 + 40) = &unk_4A23338;
  *(_QWORD *)(a1 + 152) = a6;
  v15 = *(unsigned int *)(a1 + 56);
  if ( v15 + 1 > v14 )
  {
    sub_C8D5F0(a1 + 48, (const void *)(a1 + 64), v15 + 1, 8u, v12, v13);
    v15 = *(unsigned int *)(a1 + 56);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v15) = a5;
  result = *(unsigned int *)(a5 + 24);
  v17 = *(unsigned int *)(a5 + 28);
  ++*(_DWORD *)(a1 + 56);
  if ( result + 1 > v17 )
  {
    sub_C8D5F0(a5 + 16, (const void *)(a5 + 32), result + 1, 8u, v12, v13);
    result = *(unsigned int *)(a5 + 24);
  }
  *(_QWORD *)(*(_QWORD *)(a5 + 16) + 8 * result) = v9;
  ++*(_DWORD *)(a5 + 24);
  return result;
}
