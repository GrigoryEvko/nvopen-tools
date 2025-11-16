// Function: sub_39C7DB0
// Address: 0x39c7db0
//
void __fastcall sub_39C7DB0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v6; // r14
  __int64 v7; // rdx
  __int64 v8; // rcx
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // rax
  __int64 v12; // r15
  int v13; // esi
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // r8d
  int v18; // r9d
  __int64 v19; // rcx
  unsigned __int64 v20; // r8
  int v21; // r9d
  __int64 *v22; // r12
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // r14
  _QWORD *v26; // rdi
  char *v27; // rdi
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // r15
  __int64 v32; // rbx
  __int64 v33; // rax
  __int64 v34; // rdx
  unsigned __int64 v35; // rbx
  unsigned __int64 v36; // rdi
  int v37; // [rsp+0h] [rbp-D0h]
  unsigned __int64 v38; // [rsp+8h] [rbp-C8h]
  unsigned __int64 v39; // [rsp+8h] [rbp-C8h]
  char *v40; // [rsp+10h] [rbp-C0h] BYREF
  char v41; // [rsp+20h] [rbp-B0h]
  char v42; // [rsp+21h] [rbp-AFh]
  char *v43; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v44; // [rsp+38h] [rbp-98h]
  _BYTE v45[32]; // [rsp+40h] [rbp-90h] BYREF
  __int64 v46; // [rsp+60h] [rbp-70h]
  char *v47; // [rsp+68h] [rbp-68h] BYREF
  __int64 v48; // [rsp+70h] [rbp-60h]
  _BYTE v49[88]; // [rsp+78h] [rbp-58h] BYREF

  v6 = sub_396DD80(a1[24]);
  if ( (unsigned __int16)sub_398C0A0(a1[25]) <= 4u )
    v11 = *(_QWORD *)(v6 + 160);
  else
    v11 = *(_QWORD *)(v6 + 296);
  v12 = *(_QWORD *)(v11 + 8);
  v13 = *(_DWORD *)(a3 + 8);
  v43 = v45;
  v44 = 0x200000000LL;
  if ( v13 )
    sub_39C75B0((__int64)&v43, (char **)a3, v7, v8, v9, v10);
  v14 = a1[24];
  v42 = 1;
  v40 = "debug_ranges";
  v41 = 3;
  v15 = sub_396F530(v14, (__int64)&v40);
  v47 = v49;
  v46 = v15;
  v48 = 0x200000000LL;
  if ( (_DWORD)v44 )
    sub_39C75B0((__int64)&v47, &v43, v16, (unsigned int)v44, v17, v18);
  if ( v43 != v45 )
    _libc_free((unsigned __int64)v43);
  if ( sub_39C7370((__int64)a1) )
  {
    if ( (unsigned __int16)sub_398C0A0(a1[25]) <= 4u )
      sub_39A3D70((__int64)a1, a2, 85, v46, v12);
  }
  else
  {
    sub_39A3E10((__int64)a1, a2, 85, v46, v12);
  }
  v22 = (__int64 *)a1[77];
  if ( !v22 )
    v22 = a1;
  v23 = *((unsigned int *)v22 + 186);
  v24 = *((unsigned int *)v22 + 187);
  if ( (unsigned int)v23 >= (unsigned int)v24 )
  {
    v28 = ((((((unsigned __int64)(v24 + 2) >> 1) | (v24 + 2)) >> 2) | ((unsigned __int64)(v24 + 2) >> 1) | (v24 + 2)) >> 4)
        | ((((unsigned __int64)(v24 + 2) >> 1) | (v24 + 2)) >> 2)
        | ((unsigned __int64)(v24 + 2) >> 1)
        | (v24 + 2);
    v29 = ((v28 >> 8) | v28 | (((v28 >> 8) | v28) >> 16) | (((v28 >> 8) | v28) >> 32)) + 1;
    v30 = 0xFFFFFFFFLL;
    if ( v29 <= 0xFFFFFFFF )
      v30 = v29;
    v37 = v30;
    v25 = malloc(56 * v30);
    if ( !v25 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v23 = *((unsigned int *)v22 + 186);
    }
    v31 = v22[92];
    v20 = v31 + 56 * v23;
    if ( v31 != v20 )
    {
      v32 = v25;
      do
      {
        if ( v32 )
        {
          v33 = *(_QWORD *)v31;
          *(_DWORD *)(v32 + 16) = 0;
          *(_DWORD *)(v32 + 20) = 2;
          *(_QWORD *)v32 = v33;
          *(_QWORD *)(v32 + 8) = v32 + 24;
          v34 = *(unsigned int *)(v31 + 16);
          if ( (_DWORD)v34 )
          {
            v38 = v20;
            sub_39C75B0(v32 + 8, (char **)(v31 + 8), v34, v19, v20, v21);
            v20 = v38;
          }
        }
        v31 += 56;
        v32 += 56;
      }
      while ( v20 != v31 );
      v20 = v22[92];
      v35 = v20 + 56LL * *((unsigned int *)v22 + 186);
      if ( v20 != v35 )
      {
        do
        {
          v35 -= 56LL;
          v36 = *(_QWORD *)(v35 + 8);
          if ( v36 != v35 + 24 )
          {
            v39 = v20;
            _libc_free(v36);
            v20 = v39;
          }
        }
        while ( v35 != v20 );
        v20 = v22[92];
      }
    }
    if ( (__int64 *)v20 != v22 + 94 )
      _libc_free(v20);
    v22[92] = v25;
    LODWORD(v23) = *((_DWORD *)v22 + 186);
    *((_DWORD *)v22 + 187) = v37;
  }
  else
  {
    v25 = v22[92];
  }
  v26 = (_QWORD *)(v25 + 56LL * (unsigned int)v23);
  if ( v26 )
  {
    *v26 = v46;
    v26[1] = v26 + 3;
    v26[2] = 0x200000000LL;
    if ( (_DWORD)v48 )
      sub_39C75B0((__int64)(v26 + 1), &v47, (unsigned int)v23, v19, v20, v21);
    LODWORD(v23) = *((_DWORD *)v22 + 186);
  }
  v27 = v47;
  *((_DWORD *)v22 + 186) = v23 + 1;
  if ( v27 != v49 )
    _libc_free((unsigned __int64)v27);
}
