// Function: sub_D92D30
// Address: 0xd92d30
//
__int64 __fastcall sub_D92D30(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r13
  __int64 v5; // r14
  __int64 v6; // rsi
  __int64 v7; // rsi
  __int64 v8; // rsi
  int v9; // eax
  unsigned int v10; // ebx
  unsigned int v11; // eax
  unsigned int v12; // eax
  const void *v13; // rdx
  int v14; // ecx
  bool v15; // cc
  unsigned int v16; // ecx
  int v18; // [rsp+Ch] [rbp-104h]
  const void *v19; // [rsp+20h] [rbp-F0h] BYREF
  unsigned int v20; // [rsp+28h] [rbp-E8h]
  __int64 v21; // [rsp+30h] [rbp-E0h] BYREF
  unsigned int v22; // [rsp+38h] [rbp-D8h]
  const void *v23; // [rsp+40h] [rbp-D0h] BYREF
  unsigned int v24; // [rsp+48h] [rbp-C8h]
  const void *v25; // [rsp+50h] [rbp-C0h] BYREF
  unsigned int v26; // [rsp+58h] [rbp-B8h]
  const void *v27; // [rsp+60h] [rbp-B0h] BYREF
  unsigned int v28; // [rsp+68h] [rbp-A8h]
  const void *v29; // [rsp+70h] [rbp-A0h] BYREF
  unsigned int v30; // [rsp+78h] [rbp-98h]
  __int64 v31; // [rsp+80h] [rbp-90h] BYREF
  unsigned int v32; // [rsp+88h] [rbp-88h]
  const void *v33; // [rsp+90h] [rbp-80h] BYREF
  __int64 v34; // [rsp+98h] [rbp-78h] BYREF
  unsigned int v35; // [rsp+A0h] [rbp-70h]
  const void *v36; // [rsp+A8h] [rbp-68h] BYREF
  unsigned int v37; // [rsp+B0h] [rbp-60h]
  const void *v38; // [rsp+B8h] [rbp-58h] BYREF
  unsigned int v39; // [rsp+C0h] [rbp-50h]
  const void *v40; // [rsp+C8h] [rbp-48h] BYREF
  unsigned int v41; // [rsp+D0h] [rbp-40h]

  v2 = *(_QWORD **)(a2 + 32);
  v3 = *v2;
  v4 = v2[1];
  if ( *(_WORD *)(*v2 + 24LL) )
    v3 = 0;
  if ( *(_WORD *)(v4 + 24) || (v5 = v2[2], *(_WORD *)(v5 + 24)) || !v3 )
  {
    *(_BYTE *)(a1 + 72) = 0;
    return a1;
  }
  v6 = *(_QWORD *)(v3 + 32);
  v20 = *(_DWORD *)(v6 + 32);
  if ( v20 > 0x40 )
    sub_C43780((__int64)&v19, (const void **)(v6 + 24));
  else
    v19 = *(const void **)(v6 + 24);
  v7 = *(_QWORD *)(v4 + 32);
  v22 = *(_DWORD *)(v7 + 32);
  if ( v22 > 0x40 )
    sub_C43780((__int64)&v21, (const void **)(v7 + 24));
  else
    v21 = *(_QWORD *)(v7 + 24);
  v8 = *(_QWORD *)(v5 + 32);
  v24 = *(_DWORD *)(v8 + 32);
  if ( v24 > 0x40 )
    sub_C43780((__int64)&v23, (const void **)(v8 + 24));
  else
    v23 = *(const void **)(v8 + 24);
  v9 = *(_DWORD *)(*(_QWORD *)(v3 + 32) + 32LL);
  v10 = v9 + 1;
  v18 = v9;
  sub_C44830((__int64)&v33, &v23, v9 + 1);
  if ( v24 > 0x40 && v23 )
    j_j___libc_free_0_0(v23);
  v23 = v33;
  v24 = v34;
  sub_C44830((__int64)&v33, &v21, v10);
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  v21 = (__int64)v33;
  v22 = v34;
  sub_C44830((__int64)&v33, &v19, v10);
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  v19 = v33;
  v20 = v34;
  v26 = v24;
  if ( v24 > 0x40 )
    sub_C43780((__int64)&v25, &v23);
  else
    v25 = v23;
  v32 = v22;
  if ( v22 > 0x40 )
    sub_C43780((__int64)&v31, (const void **)&v21);
  else
    v31 = v21;
  sub_C47170((__int64)&v31, 2u);
  v11 = v32;
  v32 = 0;
  LODWORD(v34) = v11;
  v33 = (const void *)v31;
  sub_C46B40((__int64)&v33, (__int64 *)&v25);
  v28 = v34;
  v27 = v33;
  if ( v32 > 0x40 && v31 )
    j_j___libc_free_0_0(v31);
  LODWORD(v34) = v20;
  if ( v20 > 0x40 )
    sub_C43780((__int64)&v33, &v19);
  else
    v33 = v19;
  sub_C47170((__int64)&v33, 2u);
  v32 = v10;
  v30 = v34;
  v29 = v33;
  if ( v10 > 0x40 )
  {
    sub_C43690((__int64)&v31, 2, 0);
    v35 = v32;
    LODWORD(v33) = v18;
    if ( v32 > 0x40 )
    {
      sub_C43780((__int64)&v34, (const void **)&v31);
      goto LABEL_33;
    }
  }
  else
  {
    v35 = v10;
    v31 = 2;
    LODWORD(v33) = v18;
  }
  v34 = v31;
LABEL_33:
  v37 = v30;
  if ( v30 > 0x40 )
    sub_C43780((__int64)&v36, &v29);
  else
    v36 = v29;
  v39 = v28;
  if ( v28 > 0x40 )
    sub_C43780((__int64)&v38, &v27);
  else
    v38 = v27;
  v12 = v26;
  v41 = v26;
  if ( v26 > 0x40 )
  {
    sub_C43780((__int64)&v40, &v25);
    v12 = v41;
    v13 = v40;
  }
  else
  {
    v13 = v25;
  }
  v14 = (int)v33;
  v15 = v32 <= 0x40;
  *(_DWORD *)(a1 + 64) = v12;
  *(_QWORD *)(a1 + 56) = v13;
  *(_DWORD *)a1 = v14;
  v16 = v35;
  *(_BYTE *)(a1 + 72) = 1;
  *(_DWORD *)(a1 + 16) = v16;
  *(_QWORD *)(a1 + 8) = v34;
  *(_DWORD *)(a1 + 32) = v37;
  *(_QWORD *)(a1 + 24) = v36;
  *(_DWORD *)(a1 + 48) = v39;
  *(_QWORD *)(a1 + 40) = v38;
  if ( !v15 && v31 )
    j_j___libc_free_0_0(v31);
  if ( v30 > 0x40 && v29 )
    j_j___libc_free_0_0(v29);
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  if ( v24 > 0x40 && v23 )
    j_j___libc_free_0_0(v23);
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  return a1;
}
