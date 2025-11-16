// Function: sub_1DBEB50
// Address: 0x1dbeb50
//
void __fastcall sub_1DBEB50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  int v6; // ebx
  __int64 v7; // r13
  int i; // eax
  int v9; // r8d
  __int64 v10; // rcx
  unsigned int *v11; // r8
  unsigned __int64 v12; // r9
  __int64 v13; // r12
  __int64 v14; // rax
  int v15; // r12d
  int v16; // r9d
  unsigned __int64 v17; // rax
  unsigned int v18; // r12d
  unsigned int v19; // r14d
  __int64 v20; // rdx
  __int64 v21; // rsi
  _QWORD *v22; // rdx
  _QWORD *v23; // rax
  const void *v24; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v26; // [rsp+30h] [rbp-80h]
  int v27; // [rsp+38h] [rbp-78h]
  unsigned int v28; // [rsp+3Ch] [rbp-74h]
  __int64 v29; // [rsp+40h] [rbp-70h] BYREF
  unsigned __int64 v30[2]; // [rsp+48h] [rbp-68h] BYREF
  _BYTE v31[32]; // [rsp+58h] [rbp-58h] BYREF
  int v32; // [rsp+78h] [rbp-38h]

  v29 = a1;
  v30[0] = (unsigned __int64)v31;
  v30[1] = 0x800000000LL;
  v32 = 0;
  sub_3945AE0(v30, 0);
  v28 = sub_1DB5A20(&v29, a2);
  if ( v28 <= 1 )
    goto LABEL_12;
  v5 = *(_QWORD *)(a1 + 240);
  v6 = 1;
  v24 = (const void *)(a1 + 416);
  v26 = *(_QWORD *)(*(_QWORD *)(v5 + 24) + 16LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
  v7 = a1;
  for ( i = sub_1E6B9A0(v5, v26, byte_3F871B3, 0); ; i = sub_1E6B9A0(*(_QWORD *)(v7 + 240), v26, byte_3F871B3, 0) )
  {
    v15 = i;
    v16 = i;
    v17 = *(unsigned int *)(v7 + 408);
    v18 = v15 & 0x7FFFFFFF;
    v19 = v18 + 1;
    if ( v18 + 1 <= (unsigned int)v17 )
      goto LABEL_3;
    v20 = v19;
    if ( v19 < v17 )
    {
      *(_DWORD *)(v7 + 408) = v19;
LABEL_3:
      v10 = *(_QWORD *)(v7 + 400);
      goto LABEL_4;
    }
    if ( v19 <= v17 )
      goto LABEL_3;
    if ( v19 > (unsigned __int64)*(unsigned int *)(v7 + 412) )
    {
      v27 = v16;
      sub_16CD150(v7 + 400, v24, v19, 8, v9, v16);
      v17 = *(unsigned int *)(v7 + 408);
      v16 = v27;
      v20 = v19;
    }
    v10 = *(_QWORD *)(v7 + 400);
    v21 = *(_QWORD *)(v7 + 416);
    v22 = (_QWORD *)(v10 + 8 * v20);
    v23 = (_QWORD *)(v10 + 8 * v17);
    if ( v22 != v23 )
    {
      do
        *v23++ = v21;
      while ( v22 != v23 );
      v10 = *(_QWORD *)(v7 + 400);
    }
    *(_DWORD *)(v7 + 408) = v19;
LABEL_4:
    *(_QWORD *)(v10 + 8LL * v18) = sub_1DBA290(v16);
    v13 = *(_QWORD *)(*(_QWORD *)(v7 + 400) + 8LL * v18);
    v14 = *(unsigned int *)(a3 + 8);
    if ( (unsigned int)v14 >= *(_DWORD *)(a3 + 12) )
    {
      sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, (int)v11, v12);
      v14 = *(unsigned int *)(a3 + 8);
    }
    ++v6;
    *(_QWORD *)(*(_QWORD *)a3 + 8 * v14) = v13;
    ++*(_DWORD *)(a3 + 8);
    if ( v28 == v6 )
      break;
  }
  sub_1DB6090(&v29, a2, *(_QWORD *)a3, *(_QWORD *)(v7 + 240), v11, v12);
LABEL_12:
  if ( (_BYTE *)v30[0] != v31 )
    _libc_free(v30[0]);
}
