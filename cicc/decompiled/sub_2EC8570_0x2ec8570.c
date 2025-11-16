// Function: sub_2EC8570
// Address: 0x2ec8570
//
unsigned __int64 __fastcall sub_2EC8570(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _DWORD *v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // r14
  unsigned __int64 v12; // r15
  _QWORD *v13; // r12
  _QWORD *v14; // rbx
  unsigned __int64 v15; // rdi
  __int64 v16; // rbx
  __int64 v17; // r12
  __int64 v18; // rbx
  unsigned __int64 result; // rax
  __int64 v20; // rdx
  __int64 i; // rdx

  v7 = *(_DWORD **)(a1 + 152);
  if ( v7 && v7[2] )
  {
    (*(void (__fastcall **)(_DWORD *))(*(_QWORD *)v7 + 8LL))(v7);
    *(_QWORD *)(a1 + 152) = 0;
  }
  v8 = *(_QWORD *)(a1 + 64);
  if ( v8 != *(_QWORD *)(a1 + 72) )
    *(_QWORD *)(a1 + 72) = v8;
  v9 = *(_QWORD *)(a1 + 128);
  if ( v9 != *(_QWORD *)(a1 + 136) )
    *(_QWORD *)(a1 + 136) = v9;
  *(_BYTE *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 172) = 0xFFFFFFFFLL;
  v10 = *(_QWORD *)(a1 + 336);
  *(_QWORD *)(a1 + 164) = 0;
  *(_QWORD *)(a1 + 180) = 0;
  *(_QWORD *)(a1 + 272) = 0;
  *(_BYTE *)(a1 + 280) = 0;
  if ( v10 != *(_QWORD *)(a1 + 344) )
    *(_QWORD *)(a1 + 344) = v10;
  v11 = *(_QWORD *)(a1 + 304);
  while ( v11 )
  {
    v12 = v11;
    v13 = (_QWORD *)(v11 + 40);
    sub_2EC3080(*(_QWORD **)(v11 + 24));
    v14 = *(_QWORD **)(v11 + 40);
    v11 = *(_QWORD *)(v11 + 16);
    while ( v14 != v13 )
    {
      v15 = (unsigned __int64)v14;
      v14 = (_QWORD *)*v14;
      j_j___libc_free_0(v15);
    }
    j_j___libc_free_0(v12);
  }
  v16 = *(unsigned int *)(a1 + 448);
  v17 = *(_QWORD *)(a1 + 440);
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 312) = a1 + 296;
  *(_QWORD *)(a1 + 320) = a1 + 296;
  *(_QWORD *)(a1 + 328) = 0;
  v18 = v17 + 16 * v16;
  *(_DWORD *)(a1 + 368) = 0;
  while ( v17 != v18 )
  {
    v18 -= 16;
    if ( *(_DWORD *)(v18 + 8) > 0x40u && *(_QWORD *)v18 )
      j_j___libc_free_0_0(*(_QWORD *)v18);
  }
  result = *(unsigned int *)(a1 + 200);
  *(_DWORD *)(a1 + 448) = 0;
  *(_DWORD *)(a1 + 712) = 0;
  if ( result != 1 )
  {
    if ( result <= 1 )
    {
      if ( !*(_DWORD *)(a1 + 204) )
      {
        sub_C8D5F0(a1 + 192, (const void *)(a1 + 208), 1u, 4u, a5, a6);
        result = *(unsigned int *)(a1 + 200);
      }
      v20 = *(_QWORD *)(a1 + 192);
      result = v20 + 4 * result;
      for ( i = v20 + 4; i != result; result += 4LL )
      {
        if ( result )
          *(_DWORD *)result = 0;
      }
    }
    *(_DWORD *)(a1 + 200) = 1;
  }
  return result;
}
