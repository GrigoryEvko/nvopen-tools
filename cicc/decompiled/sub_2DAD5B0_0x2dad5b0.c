// Function: sub_2DAD5B0
// Address: 0x2dad5b0
//
__int64 __fastcall sub_2DAD5B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 (*v4)(void); // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // r9
  void *v8; // rdi
  __int64 v9; // r13
  unsigned __int64 v10; // rax
  int v11; // edx
  int v12; // ecx
  unsigned __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned int v20; // r13d
  __int64 v22; // r13
  __int64 v23; // [rsp+8h] [rbp-28h]

  *(_QWORD *)a1 = *(_QWORD *)(a2 + 32);
  v3 = *(_QWORD *)(a2 + 16);
  v4 = *(__int64 (**)(void))(*(_QWORD *)v3 + 128LL);
  v5 = 0;
  if ( v4 != sub_2DAC790 )
  {
    v23 = *(_QWORD *)(a2 + 16);
    v5 = v4();
    v3 = v23;
  }
  *(_QWORD *)(a1 + 8) = v5;
  v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v3 + 200LL))(v3);
  v8 = *(void **)(a1 + 24);
  *(_QWORD *)(a1 + 16) = v6;
  v9 = v6;
  v10 = *(unsigned int *)(a1 + 32);
  if ( 8 * v10 )
  {
    memset(v8, 0, 8 * v10);
    v10 = *(unsigned int *)(a1 + 32);
  }
  v11 = *(_DWORD *)(v9 + 44);
  v12 = *(_DWORD *)(a1 + 88) & 0x3F;
  if ( v12 )
  {
    *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v10 - 8) &= ~(-1LL << v12);
    v10 = *(unsigned int *)(a1 + 32);
  }
  *(_DWORD *)(a1 + 88) = v11;
  v13 = (unsigned int)(v11 + 63) >> 6;
  v14 = v13;
  if ( v13 != v10 )
  {
    if ( v13 >= v10 )
    {
      v22 = v13 - v10;
      if ( v13 > *(unsigned int *)(a1 + 36) )
      {
        sub_C8D5F0(a1 + 24, (const void *)(a1 + 40), v13, 8u, v13, v7);
        v10 = *(unsigned int *)(a1 + 32);
      }
      if ( 8 * v22 )
      {
        memset((void *)(*(_QWORD *)(a1 + 24) + 8 * v10), 0, 8 * v22);
        LODWORD(v10) = *(_DWORD *)(a1 + 32);
      }
      v11 = *(_DWORD *)(a1 + 88);
      *(_DWORD *)(a1 + 32) = v22 + v10;
    }
    else
    {
      *(_DWORD *)(a1 + 32) = v13;
    }
  }
  v15 = v11 & 0x3F;
  if ( (_DWORD)v15 )
  {
    v13 = (unsigned int)v15;
    *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL * *(unsigned int *)(a1 + 32) - 8) &= ~(-1LL << v15);
  }
  v20 = sub_2DAD040((__int64 *)a1, a2, v15, v13, v14, v7);
  while ( (_BYTE)v20 && (unsigned __int8)sub_2DAD040((__int64 *)a1, a2, v16, v17, v18, v19) )
    ;
  return v20;
}
