// Function: sub_2AB95C0
// Address: 0x2ab95c0
//
unsigned __int64 __fastcall sub_2AB95C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rbx
  unsigned __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // rax
  unsigned __int64 result; // rax
  __int64 v18; // r13
  __int64 v19; // rcx
  __int64 i; // rbx
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // r8
  __int64 v24; // r12
  __int64 v25; // rax
  _QWORD *v26; // r9
  __int64 v27; // r14
  __int64 v28; // rax
  _QWORD *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // r10
  __int64 v35; // r12
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r9
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // [rsp+10h] [rbp-40h] BYREF
  __int64 v42[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = 0;
  v8 = *(_QWORD *)(a1 + 464);
  v9 = *(_QWORD *)(v8 + 8);
  if ( *(_DWORD *)(v9 + 64) == 1 )
    v6 = **(_QWORD **)(v9 + 56);
  v10 = *(_QWORD *)(a1 + 472);
  if ( *(_DWORD *)(v10 + 64) != 1 )
    BUG();
  v11 = **(_QWORD **)(v10 + 56);
  v12 = *(unsigned int *)(v11 + 88);
  if ( (_DWORD)v12 == 1 )
  {
    v13 = v11;
    v14 = v12 + 1;
    v15 = v11 + 80;
    if ( v12 + 1 <= (unsigned __int64)*(unsigned int *)(v11 + 92) )
      goto LABEL_6;
    goto LABEL_25;
  }
  v25 = sub_2BF0CC0(v8, a2);
  v26 = *(_QWORD **)(v11 + 80);
  v27 = *(_QWORD *)(a1 + 472);
  v13 = v25;
  v28 = *(unsigned int *)(v11 + 88);
  v41 = v11;
  v42[0] = v27;
  sub_2AA88F0(v26, (__int64)&v26[v28], v42);
  v29 = sub_2AA88F0(*(_QWORD **)(v27 + 56), *(_QWORD *)(v27 + 56) + 8LL * *(unsigned int *)(v27 + 64), &v41);
  v35 = ((__int64)v29 - v34) >> 3;
  if ( (_DWORD)v32 == -1 )
  {
    sub_2AB9570(v11 + 80, v13, v30, v31, v32, v33);
  }
  else
  {
    v32 = (unsigned int)v32;
    *(_QWORD *)(v33 + 8LL * (unsigned int)v32) = v13;
  }
  sub_2AB9570(v13 + 56, v11, v30, v31, v32, v33);
  sub_2AB9570(v13 + 80, v27, v36, v37, v13 + 80, v38);
  v15 = v13 + 80;
  if ( (_DWORD)v35 == -1 )
  {
    sub_2AB9570(v27 + 56, v13, v39, v40, v15, a6);
    v15 = v13 + 80;
  }
  else
  {
    *(_QWORD *)(*(_QWORD *)(v27 + 56) + 8LL * (unsigned int)v35) = v13;
  }
  v12 = *(unsigned int *)(v13 + 88);
  v14 = v12 + 1;
  if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(v13 + 92) )
  {
LABEL_25:
    sub_C8D5F0(v15, (const void *)(v13 + 96), v14, 8u, v15, a6);
    v12 = *(unsigned int *)(v13 + 88);
  }
LABEL_6:
  *(_QWORD *)(*(_QWORD *)(v13 + 80) + 8 * v12) = v6;
  ++*(_DWORD *)(v13 + 88);
  v16 = *(unsigned int *)(v6 + 64);
  if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(v6 + 68) )
  {
    sub_C8D5F0(v6 + 56, (const void *)(v6 + 72), v16 + 1, 8u, v15, a6);
    v16 = *(unsigned int *)(v6 + 64);
  }
  *(_QWORD *)(*(_QWORD *)(v6 + 56) + 8 * v16) = v13;
  ++*(_DWORD *)(v6 + 64);
  result = *(_QWORD *)(v13 + 80);
  v18 = v6 + 112;
  v19 = *(_QWORD *)(result + 8);
  *(_QWORD *)(result + 8) = *(_QWORD *)result;
  *(_QWORD *)result = v19;
  for ( i = *(_QWORD *)(v18 + 8); v18 != i; i = *(_QWORD *)(i + 8) )
  {
    while ( 1 )
    {
      if ( !i )
        BUG();
      if ( *(_BYTE *)(i - 16) == 4 && *(_BYTE *)(i + 136) == 75 )
        break;
      i = *(_QWORD *)(i + 8);
      if ( v18 == i )
        return result;
    }
    v21 = *(unsigned int *)(i + 32);
    v22 = *(_QWORD *)(i + 24);
    v23 = v21 + 1;
    v24 = *(_QWORD *)(v22 + 8);
    if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(i + 36) )
    {
      sub_C8D5F0(i + 24, (const void *)(i + 40), v21 + 1, 8u, v23, a6);
      v22 = *(_QWORD *)(i + 24);
      v21 = *(unsigned int *)(i + 32);
    }
    *(_QWORD *)(v22 + 8 * v21) = v24;
    ++*(_DWORD *)(i + 32);
    result = *(unsigned int *)(v24 + 24);
    if ( result + 1 > *(unsigned int *)(v24 + 28) )
    {
      sub_C8D5F0(v24 + 16, (const void *)(v24 + 32), result + 1, 8u, v23, a6);
      result = *(unsigned int *)(v24 + 24);
    }
    *(_QWORD *)(*(_QWORD *)(v24 + 16) + 8 * result) = i + 16;
    ++*(_DWORD *)(v24 + 24);
  }
  return result;
}
