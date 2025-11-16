// Function: sub_3119200
// Address: 0x3119200
//
void __fastcall sub_3119200(unsigned __int64 *a1, unsigned __int64 *a2, __int64 a3)
{
  unsigned __int64 *v3; // r15
  void **p_s2; // r13
  size_t v6; // r12
  size_t v7; // rcx
  _QWORD *v8; // rdi
  size_t v9; // rdx
  int v10; // eax
  __int64 v11; // r12
  unsigned __int64 v12; // r12
  __int64 v13; // rbx
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // r12
  unsigned __int64 v16; // r13
  unsigned __int64 v17; // r15
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // r8
  __int64 v21; // r9
  size_t v22; // rcx
  size_t v23; // r8
  _QWORD *v24; // rdi
  size_t v25; // rdx
  int v26; // eax
  __int64 v27; // rcx
  unsigned __int64 v28; // r8
  unsigned __int64 v29; // r12
  unsigned __int64 *v31; // [rsp+10h] [rbp-B0h]
  void **v32; // [rsp+18h] [rbp-A8h]
  size_t v33; // [rsp+18h] [rbp-A8h]
  size_t v34; // [rsp+20h] [rbp-A0h]
  __int64 v35; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v36; // [rsp+20h] [rbp-A0h]
  size_t v37; // [rsp+20h] [rbp-A0h]
  _QWORD *v38; // [rsp+28h] [rbp-98h]
  unsigned __int64 v39; // [rsp+28h] [rbp-98h]
  unsigned __int64 v40; // [rsp+28h] [rbp-98h]
  _QWORD *v41; // [rsp+28h] [rbp-98h]
  int v42; // [rsp+28h] [rbp-98h]
  int v43; // [rsp+28h] [rbp-98h]
  unsigned __int64 v44; // [rsp+28h] [rbp-98h]
  int v45; // [rsp+28h] [rbp-98h]
  int v46; // [rsp+28h] [rbp-98h]
  void *s1; // [rsp+30h] [rbp-90h] BYREF
  size_t n; // [rsp+38h] [rbp-88h]
  _QWORD v49[2]; // [rsp+40h] [rbp-80h] BYREF
  char v50; // [rsp+50h] [rbp-70h]
  void *s2; // [rsp+60h] [rbp-60h] BYREF
  size_t v52; // [rsp+68h] [rbp-58h]
  _QWORD v53[2]; // [rsp+70h] [rbp-50h] BYREF
  char v54; // [rsp+80h] [rbp-40h]

  if ( a1 == a2 )
    return;
  v3 = a1 + 1;
  if ( a2 == a1 + 1 )
    return;
  p_s2 = &s2;
  do
  {
    sub_31185E0((__int64)p_s2, a3, *(_DWORD *)(*a1 + 12));
    sub_31185E0((__int64)&s1, a3, *(_DWORD *)(*v3 + 12));
    v6 = n;
    v7 = v52;
    v8 = s1;
    v9 = v52;
    if ( n <= v52 )
      v9 = n;
    if ( !v9 || (v34 = v52, v38 = s1, v10 = memcmp(s1, s2, v9), v8 = v38, v7 = v34, !v10) )
    {
      v11 = v6 - v7;
      v10 = 0x7FFFFFFF;
      if ( v11 < 0x80000000LL )
      {
        v10 = 0x80000000;
        if ( v11 > (__int64)0xFFFFFFFF7FFFFFFFLL )
          v10 = v11;
      }
    }
    if ( !v50 || (v50 = 0, v8 == v49) )
    {
      if ( v54 )
        goto LABEL_54;
    }
    else
    {
      v45 = v10;
      j_j___libc_free_0((unsigned __int64)v8);
      v10 = v45;
      if ( v54 )
      {
LABEL_54:
        v54 = 0;
        if ( s2 != v53 )
        {
          v46 = v10;
          j_j___libc_free_0((unsigned __int64)s2);
          v10 = v46;
        }
      }
    }
    v12 = *v3;
    v31 = v3 + 1;
    if ( v10 < 0 )
    {
      *v3 = 0;
      if ( (char *)v3 - (char *)a1 > 0 )
      {
        v39 = v12;
        v32 = p_s2;
        v35 = a3;
        v13 = v3 - a1;
        do
        {
          v14 = *(v3 - 1);
          v15 = *v3--;
          *v3 = 0;
          v3[1] = v14;
          if ( v15 )
          {
            v16 = *(_QWORD *)(v15 + 24);
            if ( v16 )
            {
              sub_C7D6A0(*(_QWORD *)(v16 + 8), 16LL * *(unsigned int *)(v16 + 24), 8);
              j_j___libc_free_0(v16);
            }
            j_j___libc_free_0(v15);
          }
          --v13;
        }
        while ( v13 );
        v12 = v39;
        a3 = v35;
        p_s2 = v32;
      }
      v17 = *a1;
      *a1 = v12;
      if ( v17 )
      {
        v18 = *(_QWORD *)(v17 + 24);
        if ( v18 )
        {
          sub_C7D6A0(*(_QWORD *)(v18 + 8), 16LL * *(unsigned int *)(v18 + 24), 8);
          j_j___libc_free_0(v18);
        }
        j_j___libc_free_0(v17);
      }
      goto LABEL_26;
    }
    *v3 = 0;
    while ( 1 )
    {
      sub_31185E0((__int64)p_s2, a3, *(_DWORD *)(*(v3 - 1) + 12));
      sub_31185E0((__int64)&s1, a3, *(_DWORD *)(v12 + 12));
      v22 = n;
      v23 = v52;
      v24 = s1;
      v25 = v52;
      if ( n <= v52 )
        v25 = n;
      if ( !v25 || (v33 = v52, v37 = n, v41 = s1, v26 = memcmp(s1, s2, v25), v24 = v41, v22 = v37, v23 = v33, !v26) )
      {
        v27 = v22 - v23;
        v26 = 0x7FFFFFFF;
        if ( v27 < 0x80000000LL )
        {
          v26 = 0x80000000;
          if ( v27 > (__int64)0xFFFFFFFF7FFFFFFFLL )
            v26 = v27;
        }
      }
      if ( !v50 || (v50 = 0, v24 == v49) )
      {
        if ( !v54 )
          goto LABEL_30;
      }
      else
      {
        v42 = v26;
        j_j___libc_free_0((unsigned __int64)v24);
        v26 = v42;
        if ( !v54 )
          goto LABEL_30;
      }
      v54 = 0;
      if ( s2 != v53 )
        break;
LABEL_30:
      if ( v26 >= 0 )
        goto LABEL_48;
LABEL_31:
      v19 = *(v3 - 1);
      v20 = *v3;
      *(v3 - 1) = 0;
      *v3 = v19;
      if ( v20 )
      {
        v21 = *(_QWORD *)(v20 + 24);
        if ( v21 )
        {
          v36 = v20;
          v40 = *(_QWORD *)(v20 + 24);
          sub_C7D6A0(*(_QWORD *)(v21 + 8), 16LL * *(unsigned int *)(v21 + 24), 8);
          j_j___libc_free_0(v40);
          v20 = v36;
        }
        j_j___libc_free_0(v20);
      }
      --v3;
    }
    v43 = v26;
    j_j___libc_free_0((unsigned __int64)s2);
    if ( v43 < 0 )
      goto LABEL_31;
LABEL_48:
    v28 = *v3;
    *v3 = v12;
    if ( v28 )
    {
      v29 = *(_QWORD *)(v28 + 24);
      if ( v29 )
      {
        v44 = v28;
        sub_C7D6A0(*(_QWORD *)(v29 + 8), 16LL * *(unsigned int *)(v29 + 24), 8);
        j_j___libc_free_0(v29);
        v28 = v44;
      }
      j_j___libc_free_0(v28);
    }
LABEL_26:
    v3 = v31;
  }
  while ( a2 != v31 );
}
