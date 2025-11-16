// Function: sub_18CA960
// Address: 0x18ca960
//
_QWORD *__fastcall sub_18CA960(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v8; // edx
  __int64 v9; // rax
  __int64 *v10; // rbx
  __int64 v11; // rcx
  int v12; // edi
  __int64 v13; // r8
  __int64 v14; // r15
  _QWORD *v15; // rax
  int v16; // esi
  __int64 v17; // rcx
  _QWORD *v18; // r12
  __int64 *v19; // r10
  __int64 v20; // r11
  __int64 v21; // rdx
  __int64 v22; // rdi
  __int64 v23; // rsi
  unsigned int v24; // ecx
  __int64 *v25; // rax
  __int64 v26; // r11
  __int64 v27; // rdx
  unsigned __int64 **v28; // rax
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rax
  __int64 v31; // rdx
  __int64 *v32; // r9
  __int64 *v33; // rcx
  int v34; // esi
  __int64 v35; // rax
  unsigned int v36; // edx
  __int64 v37; // rax
  __int64 *v38; // rbx
  __int64 *v39; // r13
  __int64 v40; // rdi
  int v42; // eax
  int v43; // r9d
  __int64 v44; // [rsp+0h] [rbp-D0h]
  __int64 *v45; // [rsp+0h] [rbp-D0h]
  __int64 v46; // [rsp+8h] [rbp-C8h]
  __int64 v47; // [rsp+8h] [rbp-C8h]
  __int64 *v48; // [rsp+10h] [rbp-C0h]
  __int64 v49; // [rsp+18h] [rbp-B8h]
  __int64 v53; // [rsp+48h] [rbp-88h] BYREF
  __int64 *v54; // [rsp+50h] [rbp-80h] BYREF
  __int64 v55; // [rsp+58h] [rbp-78h]
  _BYTE v56[112]; // [rsp+60h] [rbp-70h] BYREF

  v8 = *(_DWORD *)(a6 + 16);
  v54 = (__int64 *)v56;
  v55 = 0x100000000LL;
  if ( !v8 )
  {
    v9 = *(_QWORD *)a1;
    v10 = (__int64 *)v56;
    v11 = a3;
    v12 = a3;
    v13 = 0;
    v14 = *(_QWORD *)(v9 + 24);
LABEL_3:
    v44 = v11;
    v46 = v13;
    v15 = sub_1648AB0(72, v12 + 1, v8);
    v16 = v12 + 1;
    v17 = v44;
    v18 = v15;
    if ( !v15 )
      goto LABEL_20;
    v19 = v10;
    v20 = v46;
    goto LABEL_19;
  }
  v21 = *(unsigned int *)(a6 + 24);
  v22 = *(_QWORD *)(a6 + 8);
  if ( !(_DWORD)v21 )
    goto LABEL_31;
  v23 = *(_QWORD *)(a5 + 40);
  v24 = (v21 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
  v25 = (__int64 *)(v22 + 16LL * v24);
  v26 = *v25;
  if ( v23 != *v25 )
  {
    v42 = 1;
    while ( v26 != -8 )
    {
      v43 = v42 + 1;
      v24 = (v21 - 1) & (v42 + v24);
      v25 = (__int64 *)(v22 + 16LL * v24);
      v26 = *v25;
      if ( v23 == *v25 )
        goto LABEL_7;
      v42 = v43;
    }
LABEL_31:
    v25 = (__int64 *)(v22 + 16 * v21);
  }
LABEL_7:
  v27 = v25[1];
  v28 = (unsigned __int64 **)(v27 & 0xFFFFFFFFFFFFFFF8LL);
  v29 = v27 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v27 & 4) != 0 || !v28 )
    v29 = **v28;
  v53 = sub_157ED20(v29);
  v30 = (unsigned int)*(unsigned __int8 *)(v53 + 16) - 34;
  if ( (unsigned int)v30 <= 0x36 )
  {
    v31 = 0x40018000000001LL;
    if ( _bittest64(&v31, v30) )
      sub_18CA7B0((__int64)&v54, "funclet", &v53);
  }
  v13 = (unsigned int)v55;
  v10 = v54;
  v11 = a3;
  v12 = a3;
  v8 = 16 * v55;
  v32 = &v54[7 * (unsigned int)v55];
  v14 = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
  if ( v54 == v32 )
    goto LABEL_3;
  v33 = v54;
  v34 = 0;
  do
  {
    v35 = v33[5] - v33[4];
    v33 += 7;
    v34 += v35 >> 3;
  }
  while ( v33 != v32 );
  v45 = &v54[7 * (unsigned int)v55];
  v47 = (unsigned int)v55;
  v18 = sub_1648AB0(72, (int)a3 + 1 + v34, v8);
  if ( v18 )
  {
    v19 = v10;
    v20 = v47;
    v36 = 0;
    do
    {
      v37 = v10[5] - v10[4];
      v10 += 7;
      v36 += v37 >> 3;
    }
    while ( v45 != v10 );
    v16 = v36 + a3 + 1;
    v17 = a3 + v36;
LABEL_19:
    v48 = v19;
    v49 = v20;
    sub_15F1EA0((__int64)v18, **(_QWORD **)(v14 + 16), 54, (__int64)&v18[-3 * v17 - 3], v16, a5);
    v18[7] = 0;
    sub_15F5B40((__int64)v18, v14, a1, a2, a3, a4, v48, v49);
  }
LABEL_20:
  v38 = v54;
  v39 = &v54[7 * (unsigned int)v55];
  if ( v54 != v39 )
  {
    do
    {
      v40 = *(v39 - 3);
      v39 -= 7;
      if ( v40 )
        j_j___libc_free_0(v40, v39[6] - v40);
      if ( (__int64 *)*v39 != v39 + 2 )
        j_j___libc_free_0(*v39, v39[2] + 1);
    }
    while ( v38 != v39 );
    v39 = v54;
  }
  if ( v39 != (__int64 *)v56 )
    _libc_free((unsigned __int64)v39);
  return v18;
}
