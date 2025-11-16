// Function: sub_30B2D50
// Address: 0x30b2d50
//
__int64 __fastcall sub_30B2D50(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // r13
  const void *v11; // r15
  __int64 *v12; // r13
  __int64 v13; // rax
  __int64 v14; // rsi
  int v15; // eax
  int v16; // edx
  unsigned int v17; // eax
  __int64 *v18; // rcx
  __int64 v19; // rdi
  __int64 v20; // rax
  _QWORD *v21; // rdi
  __int64 v22; // rsi
  _QWORD *v23; // rax
  int v24; // r8d
  void (__fastcall *v25)(__int64, unsigned __int64); // rax
  __int64 (__fastcall *v26)(__int64, __int64); // rax
  int v28; // ecx
  int v29; // r8d
  unsigned __int64 v31; // [rsp+10h] [rbp-50h]
  __int64 *v32; // [rsp+18h] [rbp-48h]
  __int64 v33[7]; // [rsp+28h] [rbp-38h] BYREF

  v8 = 2;
  v31 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL * *(unsigned int *)(a2 + 48) - 8);
  v9 = *(unsigned int *)(a2 + 72);
  if ( !(_DWORD)v9 )
    v8 = (*(_DWORD *)(a3 + 72) != 1) + 1;
  *(_DWORD *)(a2 + 56) = v8;
  v10 = *(unsigned int *)(a3 + 72);
  v11 = *(const void **)(a3 + 64);
  if ( v9 + v10 > (unsigned __int64)*(unsigned int *)(a2 + 76) )
  {
    sub_C8D5F0(a2 + 64, (const void *)(a2 + 80), v9 + v10, 8u, a5, a6);
    v9 = *(unsigned int *)(a2 + 72);
  }
  if ( 8 * v10 )
  {
    memcpy((void *)(*(_QWORD *)(a2 + 64) + 8 * v9), v11, 8 * v10);
    LODWORD(v9) = *(_DWORD *)(a2 + 72);
  }
  *(_DWORD *)(a2 + 72) = v10 + v9;
  v12 = *(__int64 **)(a3 + 40);
  v32 = &v12[*(unsigned int *)(a3 + 48)];
  while ( v32 != v12 )
  {
    v13 = *v12++;
    v33[0] = v13;
    sub_30B2AC0(a2 + 8, v33);
  }
  v14 = *(_QWORD *)(a2 + 16);
  v33[0] = v31;
  v15 = *(_DWORD *)(a2 + 32);
  if ( v15 )
  {
    v16 = v15 - 1;
    v17 = (v15 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
    v18 = (__int64 *)(v14 + 8LL * v17);
    v19 = *v18;
    if ( v31 == *v18 )
    {
LABEL_11:
      *v18 = -8192;
      v20 = *(unsigned int *)(a2 + 48);
      --*(_DWORD *)(a2 + 24);
      v21 = *(_QWORD **)(a2 + 40);
      ++*(_DWORD *)(a2 + 28);
      v22 = (__int64)&v21[v20];
      v23 = sub_30B02A0(v21, v22, v33);
      if ( v23 + 1 != (_QWORD *)v22 )
      {
        memmove(v23, v23 + 1, v22 - (_QWORD)(v23 + 1));
        v24 = *(_DWORD *)(a2 + 48);
      }
      *(_DWORD *)(a2 + 48) = v24 - 1;
    }
    else
    {
      v28 = 1;
      while ( v19 != -4096 )
      {
        v29 = v28 + 1;
        v17 = v16 & (v28 + v17);
        v18 = (__int64 *)(v14 + 8LL * v17);
        v19 = *v18;
        if ( v31 == *v18 )
          goto LABEL_11;
        v28 = v29;
      }
    }
  }
  v25 = *(void (__fastcall **)(__int64, unsigned __int64))(*a1 + 72LL);
  if ( v25 == sub_30B0290 )
    j_j___libc_free_0(v31);
  else
    v25((__int64)a1, v31);
  sub_30B1F70(a1[1] + 96LL, a3);
  v26 = *(__int64 (__fastcall **)(__int64, __int64))(*a1 + 80LL);
  if ( v26 == sub_30B0260 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a3 + 8LL))(a3);
  else
    return v26((__int64)a1, a3);
}
