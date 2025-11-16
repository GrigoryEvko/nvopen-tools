// Function: sub_37E6EE0
// Address: 0x37e6ee0
//
__int64 __fastcall sub_37E6EE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 *v5; // r14
  __int64 v6; // r12
  __int64 v7; // r8
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r9
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // rdi
  char *v15; // r13
  _QWORD *v16; // rsi
  __int64 v17; // rdx
  char *v18; // rcx
  __int64 v20; // [rsp+4h] [rbp-2Ch]

  v4 = sub_2E7AAE0(*(_QWORD *)(a1 + 504), a3, v20, 0);
  v5 = *(__int64 **)(a2 + 8);
  v6 = v4;
  sub_2E33BD0(*(_QWORD *)(a1 + 504) + 320LL, v4);
  v8 = *v5;
  v9 = *(_QWORD *)v6;
  *(_QWORD *)(v6 + 8) = v5;
  v10 = a1 + 200;
  v8 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v6 = v8 | v9 & 7;
  *(_QWORD *)(v8 + 8) = v6;
  *v5 = v6 | *v5 & 7;
  *(_QWORD *)(v6 + 252) = *(_QWORD *)(a2 + 252);
  *(_BYTE *)(v6 + 261) = *(_BYTE *)(a2 + 261);
  *(_BYTE *)(a2 + 261) = 0;
  v11 = *(unsigned int *)(a1 + 208);
  v12 = *(_QWORD *)(a1 + 200);
  v13 = 8LL * *(int *)(v6 + 24);
  v14 = 8 * v11;
  v15 = (char *)(v12 + v13);
  v16 = (_QWORD *)(v12 + 8 * v11);
  if ( (_QWORD *)(v12 + v13) == v16 )
  {
    if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 212) )
    {
      sub_C8D5F0(a1 + 200, (const void *)(a1 + 216), v11 + 1, 8u, v7, v10);
      v16 = (_QWORD *)(*(_QWORD *)(a1 + 200) + 8LL * *(unsigned int *)(a1 + 208));
    }
    *v16 = 0;
    ++*(_DWORD *)(a1 + 208);
    return v6;
  }
  else
  {
    LODWORD(v17) = *(_DWORD *)(a1 + 208);
    if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 212) )
    {
      sub_C8D5F0(a1 + 200, (const void *)(a1 + 216), v11 + 1, 8u, v11 + 1, v10);
      v12 = *(_QWORD *)(a1 + 200);
      v17 = *(unsigned int *)(a1 + 208);
      v14 = 8 * v17;
      v15 = (char *)(v12 + v13);
      v16 = (_QWORD *)(v12 + 8 * v17);
    }
    v18 = (char *)(v12 + v14 - 8);
    if ( v16 )
    {
      *v16 = *(_QWORD *)v18;
      v12 = *(_QWORD *)(a1 + 200);
      v17 = *(unsigned int *)(a1 + 208);
      v14 = 8 * v17;
      v18 = (char *)(v12 + 8 * v17 - 8);
    }
    if ( v15 != v18 )
    {
      memmove((void *)(v12 + v14 - (v18 - v15)), v15, v18 - v15);
      LODWORD(v17) = *(_DWORD *)(a1 + 208);
    }
    *(_DWORD *)(a1 + 208) = v17 + 1;
    *(_QWORD *)v15 = 0;
    return v6;
  }
}
