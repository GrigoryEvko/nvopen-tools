// Function: sub_30C8C60
// Address: 0x30c8c60
//
void __fastcall sub_30C8C60(_QWORD *a1, __int64 a2, __int64 a3)
{
  _BOOL4 v4; // ebx
  __int64 v5; // rax
  unsigned __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // rcx
  _QWORD **i; // rcx
  _QWORD **v11; // r14
  _QWORD *v12; // rax
  __int64 v13; // rdx
  _QWORD **v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // rax
  unsigned __int64 v18; // rdi
  unsigned __int64 *v19; // rbx
  unsigned __int64 *v20; // r12
  unsigned __int64 v21; // rdi
  __int64 v22; // rdx
  __int64 v23; // [rsp+0h] [rbp-90h] BYREF
  __int64 v24; // [rsp+8h] [rbp-88h]
  __int64 v25; // [rsp+10h] [rbp-80h]
  _QWORD *v26; // [rsp+18h] [rbp-78h]
  unsigned __int64 v27[2]; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 v28; // [rsp+30h] [rbp-60h]
  __int64 v29; // [rsp+38h] [rbp-58h]
  __int64 v30; // [rsp+40h] [rbp-50h]
  unsigned __int64 *v31; // [rsp+48h] [rbp-48h]
  __int64 v32; // [rsp+50h] [rbp-40h]
  __int64 v33; // [rsp+58h] [rbp-38h]
  __int64 v34; // [rsp+60h] [rbp-30h]
  _QWORD *v35; // [rsp+68h] [rbp-28h]

  v4 = (*(_BYTE *)(a2 + 32) & 0xFu) - 7 > 1;
  v27[0] = 0;
  v28 = 0;
  a1[2] = v4 + (unsigned int)sub_BD3960(a2);
  v5 = *(_QWORD *)(a3 + 40) - *(_QWORD *)(a3 + 32);
  a1[6] = 0;
  v29 = 0;
  a1[7] = v5 >> 3;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v27[1] = 8;
  v27[0] = sub_22077B0(0x40u);
  v6 = v27[0] + 24;
  v7 = sub_22077B0(0x200u);
  v8 = *(_QWORD *)(a3 + 32);
  v9 = *(_QWORD *)(a3 + 40);
  v31 = (unsigned __int64 *)(v27[0] + 24);
  *(_QWORD *)(v27[0] + 24) = v7;
  v35 = (_QWORD *)v6;
  v26 = (_QWORD *)v6;
  v30 = v7 + 512;
  v34 = v7 + 512;
  v25 = v7 + 512;
  v29 = v7;
  v33 = v7;
  v28 = v7;
  v32 = v7;
  v23 = v7;
  v24 = v7;
  sub_30C89C0(v27, &v23, v8, v9);
  for ( i = (_QWORD **)v28; v32 != v28; i = (_QWORD **)v28 )
  {
    v11 = (_QWORD **)*i;
    v12 = (_QWORD *)**i;
    if ( v12 )
    {
      LODWORD(v13) = 1;
      do
      {
        v12 = (_QWORD *)*v12;
        v13 = (unsigned int)(v13 + 1);
      }
      while ( v12 );
    }
    else
    {
      v13 = 1;
    }
    if ( a1[6] >= v13 )
      v13 = a1[6];
    v14 = (_QWORD **)(v30 - 8);
    a1[6] = v13;
    if ( i == v14 )
    {
      j_j___libc_free_0(v29);
      v22 = *++v31 + 512;
      v29 = *v31;
      v30 = v22;
      v28 = v29;
    }
    else
    {
      v28 = (unsigned __int64)(i + 1);
    }
    v15 = (__int64)v11[2];
    v16 = (__int64)v11[1];
    v23 = v32;
    v17 = *v35;
    v26 = v35;
    v24 = v17;
    v25 = v17 + 512;
    sub_30C89C0(v27, &v23, v16, v15);
  }
  v18 = v27[0];
  if ( v27[0] )
  {
    v19 = v31;
    v20 = v35 + 1;
    if ( v35 + 1 > v31 )
    {
      do
      {
        v21 = *v19++;
        j_j___libc_free_0(v21);
      }
      while ( v20 > v19 );
      v18 = v27[0];
    }
    j_j___libc_free_0(v18);
  }
}
