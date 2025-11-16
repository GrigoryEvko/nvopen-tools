// Function: sub_2E6F7A0
// Address: 0x2e6f7a0
//
void __fastcall sub_2E6F7A0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 *v12; // rdx
  __int64 *v13; // r12
  __int64 v14; // rdx
  unsigned int v15; // eax
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned int v19; // eax
  __int64 v20; // rbx
  _QWORD *v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rsi
  _QWORD *v24; // rax
  __int64 v25; // r8
  int v26; // r9d
  __int64 v27; // rcx
  size_t v28; // rdx
  __int64 v29; // r9
  __int64 v30; // rax
  unsigned __int64 v31; // rcx
  __int64 v32; // rdx
  __int64 v33; // rsi
  __int64 *v34; // [rsp+0h] [rbp-50h]
  __int64 v35; // [rsp+8h] [rbp-48h]
  __int64 v36[7]; // [rsp+18h] [rbp-38h] BYREF

  *(_QWORD *)(sub_2E6F1C0((__int64)a1, *(_QWORD *)(*a1 + 8LL), a3, a4, a5, a6) + 16) = a3;
  v8 = sub_2E6E010(a1, 1);
  v34 = v12;
  if ( (__int64 *)v8 != v12 )
  {
    v13 = (__int64 *)v8;
    do
    {
      v33 = *v13;
      if ( *v13 )
      {
        v14 = (unsigned int)(*(_DWORD *)(v33 + 24) + 1);
        v15 = *(_DWORD *)(v33 + 24) + 1;
      }
      else
      {
        v14 = 0;
        v15 = 0;
      }
      v16 = 0;
      if ( v15 < *(_DWORD *)(a2 + 32) )
        v16 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 8 * v14);
      v17 = *(_QWORD *)(sub_2E6F1C0((__int64)a1, v33, v14, v9, v10, v11) + 16);
      if ( v17 )
      {
        v18 = (unsigned int)(*(_DWORD *)(v17 + 24) + 1);
        v19 = *(_DWORD *)(v17 + 24) + 1;
      }
      else
      {
        v18 = 0;
        v19 = 0;
      }
      v20 = 0;
      if ( v19 < *(_DWORD *)(a2 + 32) )
        v20 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 8 * v18);
      v9 = *(_QWORD *)(v16 + 8);
      if ( v20 != v9 )
      {
        v21 = *(_QWORD **)(v9 + 24);
        v22 = *(unsigned int *)(v9 + 32);
        v35 = *(_QWORD *)(v16 + 8);
        v36[0] = v16;
        v23 = (__int64)&v21[v22];
        v24 = sub_2E6C3E0(v21, v23, v36);
        v27 = v35;
        if ( v24 + 1 != (_QWORD *)v23 )
        {
          v28 = v23 - (_QWORD)(v24 + 1);
          v23 = (__int64)(v24 + 1);
          memmove(v24, v24 + 1, v28);
          v27 = v35;
          v26 = *(_DWORD *)(v35 + 32);
        }
        v29 = (unsigned int)(v26 - 1);
        *(_DWORD *)(v27 + 32) = v29;
        *(_QWORD *)(v16 + 8) = v20;
        v30 = *(unsigned int *)(v20 + 32);
        v31 = *(unsigned int *)(v20 + 36);
        if ( v30 + 1 > v31 )
        {
          v23 = v20 + 40;
          sub_C8D5F0(v20 + 24, (const void *)(v20 + 40), v30 + 1, 8u, v25, v29);
          v30 = *(unsigned int *)(v20 + 32);
        }
        v32 = *(_QWORD *)(v20 + 24);
        *(_QWORD *)(v32 + 8 * v30) = v16;
        ++*(_DWORD *)(v20 + 32);
        sub_2E6CB70(v16, v23, v32, v31, v25, v29);
      }
      ++v13;
    }
    while ( v34 != v13 );
  }
}
