// Function: sub_EA3D20
// Address: 0xea3d20
//
_QWORD *__fastcall sub_EA3D20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v5; // r12
  const void *v6; // r14
  _BYTE *v7; // r13
  int v8; // eax
  unsigned int v9; // r8d
  __int64 *v10; // rcx
  __int64 v11; // rdx
  int v12; // r13d
  const void *v13; // r14
  _BYTE *v14; // r15
  int v15; // eax
  unsigned int v16; // r8d
  __int64 *v17; // r9
  __int64 v18; // rax
  __int64 *v19; // rdi
  _QWORD *result; // rax
  __int64 v21; // rax
  unsigned int v22; // r8d
  __int64 *v23; // r9
  __int64 v24; // rcx
  __int64 *v25; // rdx
  __int64 v26; // rax
  unsigned int v27; // r8d
  __int64 *v28; // rcx
  __int64 v29; // r15
  __int64 *v30; // rax
  __int64 *v31; // rax
  __int64 v32; // [rsp+8h] [rbp-A8h]
  __int64 *v33; // [rsp+10h] [rbp-A0h]
  __int64 *v34; // [rsp+10h] [rbp-A0h]
  unsigned int v35; // [rsp+1Ch] [rbp-94h]
  unsigned int v36; // [rsp+1Ch] [rbp-94h]
  _QWORD v37[2]; // [rsp+20h] [rbp-90h] BYREF
  _QWORD v38[2]; // [rsp+30h] [rbp-80h] BYREF
  void *v39[2]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v40; // [rsp+50h] [rbp-60h] BYREF
  void *src[2]; // [rsp+60h] [rbp-50h] BYREF
  _QWORD v42[8]; // [rsp+70h] [rbp-40h] BYREF

  v5 = (__int64 *)(a1 + 872);
  v38[0] = a2;
  v37[1] = a5;
  v38[1] = a3;
  v37[0] = a4;
  sub_C93130((__int64 *)src, (__int64)v37);
  v6 = src[0];
  v7 = src[1];
  v8 = sub_C92610();
  v9 = sub_C92740(a1 + 872, v6, (size_t)v7, v8);
  v10 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v9);
  v11 = *v10;
  if ( *v10 )
  {
    if ( v11 != -8 )
      goto LABEL_3;
    --*(_DWORD *)(a1 + 888);
  }
  v34 = v10;
  v36 = v9;
  v26 = sub_C7D670((__int64)(v7 + 17), 8);
  v27 = v36;
  v28 = v34;
  v29 = v26;
  if ( v7 )
  {
    memcpy((void *)(v26 + 16), v6, (size_t)v7);
    v27 = v36;
    v28 = v34;
  }
  v7[v29 + 16] = 0;
  *(_QWORD *)v29 = v7;
  *(_DWORD *)(v29 + 8) = 0;
  *v28 = v29;
  ++*(_DWORD *)(a1 + 884);
  v30 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v5, v27));
  v11 = *v30;
  if ( !*v30 || v11 == -8 )
  {
    v31 = v30 + 1;
    do
    {
      do
        v11 = *v31++;
      while ( !v11 );
    }
    while ( v11 == -8 );
  }
LABEL_3:
  v12 = *(_DWORD *)(v11 + 8);
  sub_C93130((__int64 *)v39, (__int64)v38);
  v13 = v39[0];
  v14 = v39[1];
  v15 = sub_C92610();
  v16 = sub_C92740((__int64)v5, v13, (size_t)v14, v15);
  v17 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v16);
  v18 = *v17;
  if ( *v17 )
  {
    if ( v18 != -8 )
      goto LABEL_5;
    --*(_DWORD *)(a1 + 888);
  }
  v33 = v17;
  v35 = v16;
  v21 = sub_C7D670((__int64)(v14 + 17), 8);
  v22 = v35;
  v23 = v33;
  v24 = v21;
  if ( v14 )
  {
    v32 = v21;
    memcpy((void *)(v21 + 16), v13, (size_t)v14);
    v22 = v35;
    v23 = v33;
    v24 = v32;
  }
  v14[v24 + 16] = 0;
  *(_QWORD *)v24 = v14;
  *(_DWORD *)(v24 + 8) = 0;
  *v23 = v24;
  ++*(_DWORD *)(a1 + 884);
  v25 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v5, v22));
  v18 = *v25;
  if ( *v25 )
    goto LABEL_15;
  do
  {
    do
    {
      v18 = v25[1];
      ++v25;
    }
    while ( !v18 );
LABEL_15:
    ;
  }
  while ( v18 == -8 );
LABEL_5:
  v19 = (__int64 *)v39[0];
  *(_DWORD *)(v18 + 8) = v12;
  if ( v19 != &v40 )
    j_j___libc_free_0(v19, v40 + 1);
  result = v42;
  if ( src[0] != v42 )
    return (_QWORD *)j_j___libc_free_0(src[0], v42[0] + 1LL);
  return result;
}
