// Function: sub_E6D970
// Address: 0xe6d970
//
unsigned __int64 __fastcall sub_E6D970(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        void *a4,
        void *a5,
        int a6,
        int a7,
        char a8,
        _BYTE *a9)
{
  const void *v9; // r13
  _BYTE *v10; // rbx
  int v11; // eax
  unsigned int v12; // r8d
  __int64 *v13; // r9
  __int64 v14; // rax
  unsigned __int64 v15; // r14
  __int64 v16; // rax
  unsigned int v17; // r8d
  __int64 *v18; // r9
  __int64 v19; // rcx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r13
  __int64 v23; // rax
  unsigned __int64 *v24; // rcx
  int v25; // r8d
  bool v26; // cf
  unsigned __int64 v27; // rax
  int v28; // r15d
  __int64 v29; // rax
  __int64 v31; // rax
  _QWORD *v32; // [rsp+0h] [rbp-E0h]
  __int64 *v33; // [rsp+8h] [rbp-D8h]
  unsigned int v34; // [rsp+10h] [rbp-D0h]
  int v37; // [rsp+18h] [rbp-C8h]
  int v38; // [rsp+20h] [rbp-C0h]
  __int64 *v39; // [rsp+28h] [rbp-B8h]
  void *src[2]; // [rsp+30h] [rbp-B0h] BYREF
  _QWORD v41[2]; // [rsp+40h] [rbp-A0h] BYREF
  _QWORD v42[2]; // [rsp+50h] [rbp-90h] BYREF
  char v43; // [rsp+60h] [rbp-80h]
  __int16 v44; // [rsp+70h] [rbp-70h]
  void *v45[4]; // [rsp+80h] [rbp-60h] BYREF
  __int16 v46; // [rsp+A0h] [rbp-40h]

  v38 = a3;
  v42[0] = a2;
  v42[1] = a3;
  v45[2] = a4;
  v45[3] = a5;
  v44 = 2053;
  v46 = 1282;
  v45[0] = v42;
  v43 = 44;
  sub_CA0F50((__int64 *)src, v45);
  v9 = src[0];
  v10 = src[1];
  v11 = sub_C92610();
  v12 = sub_C92740(a1 + 1976, v9, (size_t)v10, v11);
  v13 = (__int64 *)(*(_QWORD *)(a1 + 1976) + 8LL * v12);
  v14 = *v13;
  if ( *v13 )
  {
    if ( v14 != -8 )
    {
      v39 = (__int64 *)(*(_QWORD *)(a1 + 1976) + 8LL * v12);
      if ( src[0] != v41 )
      {
        j_j___libc_free_0(src[0], v41[0] + 1LL);
        v14 = *v39;
      }
      return *(_QWORD *)(v14 + 8);
    }
    --*(_DWORD *)(a1 + 1992);
  }
  v33 = v13;
  v34 = v12;
  v16 = sub_C7D670((__int64)(v10 + 17), 8);
  v17 = v34;
  v18 = v33;
  v19 = v16;
  if ( v10 )
  {
    v32 = (_QWORD *)v16;
    memcpy((void *)(v16 + 16), v9, (size_t)v10);
    v17 = v34;
    v18 = v33;
    v19 = (__int64)v32;
  }
  v10[v19 + 16] = 0;
  *(_QWORD *)v19 = v10;
  *(_QWORD *)(v19 + 8) = 0;
  *v18 = v19;
  ++*(_DWORD *)(a1 + 1988);
  v22 = *(_QWORD *)(a1 + 1976) + 8LL * (unsigned int)sub_C929D0((__int64 *)(a1 + 1976), v17);
  if ( *(_QWORD *)v22 == -8 || !*(_QWORD *)v22 )
  {
    do
    {
      do
      {
        v23 = *(_QWORD *)(v22 + 8);
        v22 += 8;
      }
      while ( !v23 );
    }
    while ( v23 == -8 );
  }
  if ( src[0] != v41 )
    j_j___libc_free_0(src[0], v41[0] + 1LL);
  if ( a9 )
  {
    v46 = 257;
    if ( *a9 )
    {
      v45[0] = a9;
      LOBYTE(v46) = 3;
    }
    a9 = (_BYTE *)sub_E6C380(a1, (__int64 *)v45, 0, v20, v21);
  }
  v24 = *(unsigned __int64 **)v22;
  v25 = 0;
  v27 = **(_QWORD **)v22;
  v26 = v27 < (unsigned __int64)a5;
  LODWORD(v27) = v27 - (_DWORD)a5;
  if ( v26 )
    v27 = **(_QWORD **)v22;
  else
    v25 = (int)a5;
  *(_QWORD *)(a1 + 752) += 192LL;
  v28 = (_DWORD)v24 + v27 + 16;
  v29 = *(_QWORD *)(a1 + 672);
  v15 = (v29 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_QWORD *)(a1 + 680) >= v15 + 192 && v29 )
  {
    *(_QWORD *)(a1 + 672) = v15 + 192;
  }
  else
  {
    v37 = v25;
    v31 = sub_9D1E70(a1 + 672, 192, 192, 3);
    v25 = v37;
    v15 = v31;
  }
  sub_E95020(v15, a2, v38, v28, v25, a6, a7, a8, (__int64)a9);
  *(_QWORD *)(*(_QWORD *)v22 + 8LL) = v15;
  sub_E6B260((_QWORD *)a1, v15);
  return v15;
}
