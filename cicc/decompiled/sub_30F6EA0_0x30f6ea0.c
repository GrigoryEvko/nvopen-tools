// Function: sub_30F6EA0
// Address: 0x30f6ea0
//
void __fastcall sub_30F6EA0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  unsigned __int64 v8; // rbx
  void *v11; // rdi
  __int64 v14; // r9
  unsigned int v15; // r10d
  int v16; // eax
  __int64 v17; // rax
  char **v18; // r8
  char **v19; // r13
  char *v20; // r14
  unsigned int v21; // eax
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  unsigned __int64 v25; // rax
  char **v26; // rdx
  size_t v27; // rdx
  unsigned __int64 v28; // [rsp+0h] [rbp-50h]
  __int64 v29; // [rsp+0h] [rbp-50h]
  __int64 v30; // [rsp+8h] [rbp-48h]
  __int64 v31; // [rsp+8h] [rbp-48h]
  __int64 v32; // [rsp+10h] [rbp-40h]
  __int64 v33; // [rsp+10h] [rbp-40h]
  char **v34; // [rsp+18h] [rbp-38h]
  unsigned int v35; // [rsp+18h] [rbp-38h]
  unsigned int v36; // [rsp+18h] [rbp-38h]

  v11 = (void *)(a1 + 16);
  *(_QWORD *)a1 = v11;
  v14 = a7;
  *(_QWORD *)(a1 + 8) = 0x800000000LL;
  v15 = *(_DWORD *)(a2 + 8);
  if ( v15 && a1 != a2 )
  {
    v27 = 8LL * v15;
    if ( v15 <= 8 )
      goto LABEL_13;
    v31 = a5;
    v33 = a3;
    v36 = *(_DWORD *)(a2 + 8);
    sub_C8D5F0(a1, v11, v15, 8u, a5, a7);
    v11 = *(void **)a1;
    v15 = v36;
    a3 = v33;
    v27 = 8LL * *(unsigned int *)(a2 + 8);
    a5 = v31;
    v14 = a7;
    if ( v27 )
    {
LABEL_13:
      v29 = v14;
      v30 = a5;
      v32 = a3;
      v35 = v15;
      memcpy(v11, *(const void **)a2, v27);
      v14 = v29;
      a5 = v30;
      a3 = v32;
      v15 = v35;
    }
    *(_DWORD *)(a1 + 8) = v15;
  }
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 88) = 0x300000000LL;
  *(_QWORD *)(a1 + 152) = 0x300000000LL;
  v16 = qword_50313A8;
  *(_QWORD *)(a1 + 144) = a1 + 160;
  if ( BYTE4(a8) )
    v16 = a8;
  *(_DWORD *)(a1 + 232) = v16;
  *(_BYTE *)(a1 + 236) = 1;
  v17 = *(unsigned int *)(a2 + 8);
  *(_QWORD *)(a1 + 256) = a5;
  v18 = *(char ***)a2;
  *(_QWORD *)(a1 + 240) = a3;
  v19 = v18;
  *(_QWORD *)(a1 + 248) = a4;
  *(_QWORD *)(a1 + 264) = a6;
  *(_QWORD *)(a1 + 272) = v14;
  v34 = &v18[v17];
  if ( v34 != v18 )
  {
    do
    {
      v20 = *v19;
      v21 = sub_DCF980(a4, *v19);
      v24 = *(unsigned int *)(a1 + 88);
      if ( !v21 )
        v21 = qword_5031488;
      v25 = v8 & 0xFFFFFFFF00000000LL | v21;
      v8 = v25;
      if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 92) )
      {
        v28 = v25;
        sub_C8D5F0(a1 + 80, (const void *)(a1 + 96), v24 + 1, 0x10u, v22, v23);
        v24 = *(unsigned int *)(a1 + 88);
        v25 = v28;
      }
      v26 = (char **)(*(_QWORD *)(a1 + 80) + 16 * v24);
      ++v19;
      *v26 = v20;
      v26[1] = (char *)v25;
      ++*(_DWORD *)(a1 + 88);
    }
    while ( v34 != v19 );
  }
  sub_30F6A70(a1);
}
