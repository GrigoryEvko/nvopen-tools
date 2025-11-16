// Function: sub_23072C0
// Address: 0x23072c0
//
_QWORD *__fastcall sub_23072C0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  _QWORD *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  _QWORD *v13; // r13
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // r14
  __int64 v19; // r15
  unsigned __int64 v20; // rdi
  char *v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 v24; // r14
  __int64 v25; // r13
  unsigned __int64 v26; // rdi
  __int64 v28; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v29; // [rsp+8h] [rbp-A8h]
  __int64 v30; // [rsp+10h] [rbp-A0h]
  unsigned int v31; // [rsp+18h] [rbp-98h]
  char *v32; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v33; // [rsp+28h] [rbp-88h]
  char v34; // [rsp+30h] [rbp-80h] BYREF
  __int64 v35; // [rsp+40h] [rbp-70h]
  __int64 v36; // [rsp+48h] [rbp-68h]
  __int64 v37; // [rsp+50h] [rbp-60h]
  unsigned int v38; // [rsp+58h] [rbp-58h]
  char *v39; // [rsp+60h] [rbp-50h] BYREF
  __int64 v40; // [rsp+68h] [rbp-48h]
  _BYTE v41[64]; // [rsp+70h] [rbp-40h] BYREF

  sub_22A5EA0((__int64)&v28, a2 + 8, a3, a4);
  v35 = 1;
  ++v28;
  v36 = v29;
  v29 = 0;
  v37 = v30;
  v39 = v41;
  v30 = 0;
  v38 = v31;
  v31 = 0;
  v40 = 0x100000000LL;
  if ( v33 )
    sub_2303CE0((__int64)&v39, &v32, v33, v5, v6, v7);
  v8 = (_QWORD *)sub_22077B0(0x40u);
  v13 = v8;
  if ( v8 )
  {
    ++v35;
    v8[1] = 1;
    *v8 = &unk_4A0B060;
    v14 = v36;
    v36 = 0;
    v13[2] = v14;
    v15 = v37;
    v37 = 0;
    v13[3] = v15;
    LODWORD(v15) = v38;
    v38 = 0;
    *((_DWORD *)v13 + 8) = v15;
    v13[5] = v13 + 7;
    v13[6] = 0x100000000LL;
    if ( (_DWORD)v40 )
      sub_2303CE0((__int64)(v13 + 5), &v39, v9, v10, v11, v12);
  }
  if ( v39 != v41 )
    _libc_free((unsigned __int64)v39);
  v16 = v38;
  if ( v38 )
  {
    v17 = v36;
    v18 = v36 + 56LL * v38;
    do
    {
      v19 = v17 + 56;
      if ( *(_QWORD *)v17 != -4096 && *(_QWORD *)v17 != -8192 )
      {
        v20 = *(_QWORD *)(v17 + 40);
        if ( v20 != v19 )
          _libc_free(v20);
        sub_C7D6A0(*(_QWORD *)(v17 + 16), 8LL * *(unsigned int *)(v17 + 32), 8);
      }
      v17 += 56;
    }
    while ( v18 != v19 );
    v16 = v38;
  }
  sub_C7D6A0(v36, 56 * v16, 8);
  v21 = v32;
  *a1 = v13;
  if ( v21 != &v34 )
    _libc_free((unsigned __int64)v21);
  v22 = v31;
  if ( v31 )
  {
    v23 = v29;
    v24 = v29 + 56LL * v31;
    do
    {
      v25 = v23 + 56;
      if ( *(_QWORD *)v23 != -8192 && *(_QWORD *)v23 != -4096 )
      {
        v26 = *(_QWORD *)(v23 + 40);
        if ( v26 != v25 )
          _libc_free(v26);
        sub_C7D6A0(*(_QWORD *)(v23 + 16), 8LL * *(unsigned int *)(v23 + 32), 8);
      }
      v23 += 56;
    }
    while ( v24 != v25 );
    v22 = v31;
  }
  sub_C7D6A0(v29, 56 * v22, 8);
  return a1;
}
