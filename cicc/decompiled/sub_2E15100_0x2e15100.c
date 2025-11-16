// Function: sub_2E15100
// Address: 0x2e15100
//
void __fastcall sub_2E15100(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // r12d
  __int64 v6; // rsi
  __int64 *v7; // r15
  __int64 v8; // rax
  unsigned int *v9; // r8
  __int64 v10; // r9
  __int64 v11; // r13
  __int64 v12; // rax
  int v13; // eax
  __int64 v14; // r8
  unsigned __int64 v15; // rcx
  int v16; // r13d
  unsigned int v17; // eax
  unsigned int v18; // edx
  __int64 v19; // r9
  unsigned __int64 v20; // r15
  __int64 *v21; // rdx
  __int64 *v22; // rdi
  __int64 v23; // [rsp+8h] [rbp-A8h]
  unsigned int v24; // [rsp+14h] [rbp-9Ch]
  const void *v25; // [rsp+18h] [rbp-98h]
  unsigned int v27; // [rsp+38h] [rbp-78h]
  unsigned int v28; // [rsp+3Ch] [rbp-74h]
  __int64 v29; // [rsp+40h] [rbp-70h] BYREF
  unsigned __int64 v30[2]; // [rsp+48h] [rbp-68h] BYREF
  _BYTE v31[32]; // [rsp+58h] [rbp-58h] BYREF
  int v32; // [rsp+78h] [rbp-38h]

  v29 = a1;
  v30[0] = (unsigned __int64)v31;
  v30[1] = 0x800000000LL;
  v32 = 0;
  sub_3157150(v30, 0);
  v28 = sub_2E0BE90(&v29, a2);
  if ( v28 <= 1 )
    goto LABEL_12;
  v4 = 1;
  v25 = (const void *)(a1 + 168);
  v27 = *(_DWORD *)(a2 + 112);
  do
  {
    v13 = sub_2EC0780(*(_QWORD *)(a1 + 8), v27, byte_3F871B3, 0);
    v15 = *(unsigned int *)(a1 + 160);
    v16 = v13;
    v17 = v13 & 0x7FFFFFFF;
    v18 = v17 + 1;
    if ( v17 + 1 <= (unsigned int)v15 || v18 == v15 )
      goto LABEL_3;
    if ( v18 < v15 )
    {
      *(_DWORD *)(a1 + 160) = v18;
LABEL_3:
      v6 = *(_QWORD *)(a1 + 152);
      goto LABEL_4;
    }
    v19 = *(_QWORD *)(a1 + 168);
    v20 = v18 - v15;
    if ( v18 > (unsigned __int64)*(unsigned int *)(a1 + 164) )
    {
      v23 = *(_QWORD *)(a1 + 168);
      v24 = v17;
      sub_C8D5F0(a1 + 152, v25, v18, 8u, v14, v19);
      v15 = *(unsigned int *)(a1 + 160);
      v19 = v23;
      v17 = v24;
    }
    v6 = *(_QWORD *)(a1 + 152);
    v21 = (__int64 *)(v6 + 8 * v15);
    v22 = &v21[v20];
    if ( v21 != v22 )
    {
      do
        *v21++ = v19;
      while ( v22 != v21 );
      LODWORD(v15) = *(_DWORD *)(a1 + 160);
      v6 = *(_QWORD *)(a1 + 152);
    }
    *(_DWORD *)(a1 + 160) = v20 + v15;
LABEL_4:
    v7 = (__int64 *)(v6 + 8LL * v17);
    v8 = sub_2E10F30(v16);
    *v7 = v8;
    v11 = v8;
    v12 = *(unsigned int *)(a3 + 8);
    if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
      sub_C8D5F0(a3, (const void *)(a3 + 16), v12 + 1, 8u, (__int64)v9, v10);
      v12 = *(unsigned int *)(a3 + 8);
    }
    ++v4;
    *(_QWORD *)(*(_QWORD *)a3 + 8 * v12) = v11;
    ++*(_DWORD *)(a3 + 8);
  }
  while ( v28 != v4 );
  sub_2E0C920(&v29, a2, *(_QWORD *)a3, *(_QWORD *)(a1 + 8), v9, v10);
LABEL_12:
  if ( (_BYTE *)v30[0] != v31 )
    _libc_free(v30[0]);
}
