// Function: sub_D38F70
// Address: 0xd38f70
//
__int64 __fastcall sub_D38F70(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // r12
  _QWORD *v10; // rax
  __int64 v11; // rdx
  _BOOL8 v12; // rdi
  __int64 *v13; // rax
  __int64 *v14; // rbx
  int v15; // r12d
  __int64 v16; // r15
  _BOOL8 v17; // rdi
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rax
  __int64 v21; // r14
  _QWORD *v22; // rax
  __int64 v23; // rdx
  __int64 v25; // rax
  _DWORD *v26; // rax
  __int64 i; // rdx
  int *v28; // rdi
  __int64 j; // rbx
  __int64 *v31; // [rsp+18h] [rbp-A8h]
  __int64 v34; // [rsp+38h] [rbp-88h]
  unsigned __int8 v35; // [rsp+46h] [rbp-7Ah]
  unsigned __int8 v36; // [rsp+47h] [rbp-79h]
  int v37; // [rsp+58h] [rbp-68h]
  _BYTE v38[8]; // [rsp+60h] [rbp-60h] BYREF
  int v39; // [rsp+68h] [rbp-58h] BYREF
  __int64 v40; // [rsp+70h] [rbp-50h]
  int *v41; // [rsp+78h] [rbp-48h]
  int *v42; // [rsp+80h] [rbp-40h]
  __int64 v43; // [rsp+88h] [rbp-38h]

  v34 = *a1;
  v39 = 0;
  v40 = 0;
  v41 = &v39;
  v42 = &v39;
  v43 = 0;
  v8 = sub_22077B0(48);
  *(_QWORD *)(v8 + 32) = 0;
  v9 = v8;
  *(_DWORD *)(v8 + 40) = 0;
  v10 = sub_D38ED0((__int64)v38, (__int64 *)(v8 + 32));
  if ( v11 )
  {
    v12 = v10 || (int *)v11 == &v39 || *(_QWORD *)(v11 + 32) > 0LL;
    sub_220F040(v12, v9, v11, &v39);
    ++v43;
  }
  else
  {
    j_j___libc_free_0(v9, 48);
  }
  v13 = &a1[a2];
  v14 = a1 + 1;
  v31 = v13;
  if ( v13 == v14 )
  {
    v36 = 1;
    *(_DWORD *)(a6 + 8) = 0;
  }
  else
  {
    v36 = 1;
    v15 = 1;
    v16 = a3;
    do
    {
      v20 = sub_D35010(v16, v34, v16, *v14, a4, a5, 1, 1);
      v37 = v20;
      v35 = BYTE4(v20);
      if ( !BYTE4(v20) )
        goto LABEL_15;
      v21 = sub_22077B0(48);
      *(_DWORD *)(v21 + 40) = v15;
      *(_QWORD *)(v21 + 32) = v37;
      v22 = sub_D38ED0((__int64)v38, (__int64 *)(v21 + 32));
      if ( !v23 )
      {
        j_j___libc_free_0(v21, 48);
LABEL_15:
        v36 = 0;
        goto LABEL_16;
      }
      v17 = v22 || (int *)v23 == &v39 || v37 < *(_QWORD *)(v23 + 32);
      sub_220F040(v17, v21, v23, &v39);
      ++v43;
      ++v15;
      v36 &= &v39 == (int *)sub_220EF30(v21);
      ++v14;
    }
    while ( v31 != v14 );
    *(_DWORD *)(a6 + 8) = 0;
    if ( !v36 )
    {
      if ( a2 )
      {
        v25 = 0;
        if ( a2 > *(unsigned int *)(a6 + 12) )
        {
          sub_C8D5F0(a6, (const void *)(a6 + 16), a2, 4u, v18, v19);
          v25 = 4LL * *(unsigned int *)(a6 + 8);
        }
        v26 = (_DWORD *)(*(_QWORD *)a6 + v25);
        for ( i = *(_QWORD *)a6 + 4 * a2; (_DWORD *)i != v26; ++v26 )
        {
          if ( v26 )
            *v26 = 0;
        }
        *(_DWORD *)(a6 + 8) = a2;
      }
      v28 = v41;
      for ( j = 0; v28 != &v39; v28 = (int *)sub_220EF30(v28) )
      {
        *(_DWORD *)(*(_QWORD *)a6 + j) = v28[10];
        j += 4;
      }
      v36 = v35;
    }
  }
LABEL_16:
  sub_D32950(v40);
  return v36;
}
