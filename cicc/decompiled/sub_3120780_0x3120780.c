// Function: sub_3120780
// Address: 0x3120780
//
void __fastcall sub_3120780(__int64 a1, __int64 a2)
{
  unsigned int v3; // r15d
  int v4; // r12d
  __int64 *v5; // r15
  __int64 *v6; // r13
  __int64 v7; // rax
  __int64 v8; // rdi
  _BYTE *v9; // rax
  char *v10; // r15
  char **v11; // r13
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  char *v16; // rdi
  char *v17; // r13
  char *v18; // r15
  int v19; // eax
  unsigned __int64 *v20; // rbx
  unsigned __int64 *v21; // r12
  unsigned int v22; // r13d
  int v23; // r12d
  _BYTE *v24; // rax
  __int64 *v25; // r12
  __int64 v26; // r13
  __int64 v27; // r14
  __int64 i; // r13
  char *v29; // [rsp+10h] [rbp-100h]
  char ***v30; // [rsp+18h] [rbp-F8h]
  __int64 v31; // [rsp+28h] [rbp-E8h] BYREF
  __int64 *v32; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v33; // [rsp+38h] [rbp-D8h]
  _BYTE v34[32]; // [rsp+40h] [rbp-D0h] BYREF
  char *v35; // [rsp+60h] [rbp-B0h] BYREF
  int v36; // [rsp+68h] [rbp-A8h]
  char v37; // [rsp+70h] [rbp-A0h] BYREF
  char *v38; // [rsp+A0h] [rbp-70h] BYREF
  int v39; // [rsp+A8h] [rbp-68h]
  char v40; // [rsp+B0h] [rbp-60h] BYREF

  v3 = *(_DWORD *)(a2 + 40);
  v32 = (__int64 *)v34;
  v33 = 0x100000000LL;
  if ( v3 )
  {
    v25 = (__int64 *)v34;
    v26 = 1;
    if ( v3 != 1 )
    {
      sub_95D880((__int64)&v32, v3);
      v25 = v32;
      v26 = *(unsigned int *)(a2 + 40);
    }
    v27 = *(_QWORD *)(a2 + 32);
    for ( i = v27 + 32 * v26; i != v27; v25 += 4 )
    {
      if ( v25 )
      {
        *v25 = (__int64)(v25 + 2);
        sub_311CD80(v25, *(_BYTE **)v27, *(_QWORD *)v27 + *(_QWORD *)(v27 + 8));
      }
      v27 += 32;
    }
    LODWORD(v33) = v3;
  }
  LODWORD(v38) = v3;
  v4 = 4;
  sub_CB6200(a1, (unsigned __int8 *)&v38, 4u);
  v5 = v32;
  v6 = &v32[4 * (unsigned int)v33];
  if ( v32 != v6 )
  {
    do
    {
      v8 = sub_CB6200(a1, (unsigned __int8 *)*v5, v5[1]);
      v9 = *(_BYTE **)(v8 + 32);
      if ( (unsigned __int64)v9 < *(_QWORD *)(v8 + 24) )
      {
        *(_QWORD *)(v8 + 32) = v9 + 1;
        *v9 = 0;
      }
      else
      {
        sub_CB5D20(v8, 0);
      }
      v7 = v5[1];
      v5 += 4;
      v4 += v7 + 1;
    }
    while ( v6 != v5 );
    v22 = ((v4 + 3) & 0xFFFFFFFC) - v4;
    if ( ((v4 + 3) & 0xFFFFFFFC) != v4 )
    {
      v23 = 0;
      do
      {
        v24 = *(_BYTE **)(a1 + 32);
        if ( (unsigned __int64)v24 < *(_QWORD *)(a1 + 24) )
        {
          *(_QWORD *)(a1 + 32) = v24 + 1;
          *v24 = 0;
        }
        else
        {
          sub_CB5D20(a1, 0);
        }
        ++v23;
      }
      while ( v22 != v23 );
    }
  }
  sub_311ECE0((__int64)&v35, a2);
  LODWORD(v38) = v36;
  sub_CB6200(a1, (unsigned __int8 *)&v38, 4u);
  v10 = v35;
  v29 = &v35[8 * v36];
  if ( v29 != v35 )
  {
    v30 = (char ***)v35;
    do
    {
      v11 = *v30;
      v38 = **v30;
      sub_CB6200(a1, (unsigned __int8 *)&v38, 8u);
      LODWORD(v38) = *((_DWORD *)v11 + 2);
      sub_CB6200(a1, (unsigned __int8 *)&v38, 4u);
      LODWORD(v38) = *((_DWORD *)v11 + 3);
      sub_CB6200(a1, (unsigned __int8 *)&v38, 4u);
      LODWORD(v38) = *((_DWORD *)v11 + 4);
      sub_CB6200(a1, (unsigned __int8 *)&v38, 4u);
      sub_3120550((__int64)&v38, (__int64)v11, v12, v13, v14, v15);
      LODWORD(v31) = v39;
      sub_CB6200(a1, (unsigned __int8 *)&v31, 4u);
      v16 = v38;
      v17 = &v38[16 * v39];
      if ( v17 != v38 )
      {
        v18 = v38;
        do
        {
          v19 = *(_DWORD *)v18;
          v18 += 16;
          LODWORD(v31) = v19;
          sub_CB6200(a1, (unsigned __int8 *)&v31, 4u);
          LODWORD(v31) = *((_DWORD *)v18 - 3);
          sub_CB6200(a1, (unsigned __int8 *)&v31, 4u);
          v31 = *((_QWORD *)v18 - 1);
          sub_CB6200(a1, (unsigned __int8 *)&v31, 8u);
        }
        while ( v17 != v18 );
        v16 = v38;
      }
      if ( v16 != &v40 )
        _libc_free((unsigned __int64)v16);
      ++v30;
    }
    while ( v29 != (char *)v30 );
    v10 = v35;
  }
  if ( v10 != &v37 )
    _libc_free((unsigned __int64)v10);
  v20 = (unsigned __int64 *)v32;
  v21 = (unsigned __int64 *)&v32[4 * (unsigned int)v33];
  if ( v32 != (__int64 *)v21 )
  {
    do
    {
      v21 -= 4;
      if ( (unsigned __int64 *)*v21 != v21 + 2 )
        j_j___libc_free_0(*v21);
    }
    while ( v20 != v21 );
    v21 = (unsigned __int64 *)v32;
  }
  if ( v21 != (unsigned __int64 *)v34 )
    _libc_free((unsigned __int64)v21);
}
