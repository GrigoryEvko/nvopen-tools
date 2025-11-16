// Function: sub_264FAB0
// Address: 0x264fab0
//
void __fastcall sub_264FAB0(__int64 a1, __int64 a2)
{
  unsigned int v3; // esi
  __int64 v4; // r8
  __int64 v5; // rdx
  int v6; // r11d
  __int64 *v7; // r10
  unsigned int v8; // eax
  __int64 *v9; // rdi
  __int64 v10; // rcx
  int v11; // eax
  int v12; // ecx
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 *v15; // r12
  __int64 *v16; // r14
  __int64 v17; // rdi
  __int64 *v18; // r13
  unsigned __int64 v19; // r12
  __int64 v20; // rdi
  unsigned __int64 v21; // rbx
  volatile signed __int32 *v22; // rdi
  __int64 v23; // [rsp+8h] [rbp-48h] BYREF
  __int64 *v24; // [rsp+10h] [rbp-40h] BYREF
  unsigned __int64 v25; // [rsp+18h] [rbp-38h]

  v3 = *(_DWORD *)(a2 + 24);
  v23 = a1;
  if ( !v3 )
  {
    ++*(_QWORD *)a2;
    v24 = 0;
LABEL_32:
    sub_2646220(a2, 2 * v3);
LABEL_33:
    sub_263DCA0(a2, &v23, &v24);
    v5 = v23;
    v7 = v24;
    v12 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_13;
  }
  v4 = *(_QWORD *)(a2 + 8);
  v5 = a1;
  v6 = 1;
  v7 = 0;
  v8 = (v3 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v9 = (__int64 *)(v4 + 8LL * v8);
  v10 = *v9;
  if ( *v9 == v5 )
    return;
  while ( v10 != -4096 )
  {
    if ( v7 || v10 != -8192 )
      v9 = v7;
    v8 = (v3 - 1) & (v6 + v8);
    v10 = *(_QWORD *)(v4 + 8LL * v8);
    if ( v5 == v10 )
      return;
    ++v6;
    v7 = v9;
    v9 = (__int64 *)(v4 + 8LL * v8);
  }
  v11 = *(_DWORD *)(a2 + 16);
  if ( !v7 )
    v7 = v9;
  ++*(_QWORD *)a2;
  v12 = v11 + 1;
  v24 = v7;
  if ( 4 * (v11 + 1) >= 3 * v3 )
    goto LABEL_32;
  if ( v3 - *(_DWORD *)(a2 + 20) - v12 <= v3 >> 3 )
  {
    sub_2646220(a2, v3);
    goto LABEL_33;
  }
LABEL_13:
  *(_DWORD *)(a2 + 16) = v12;
  if ( *v7 != -4096 )
    --*(_DWORD *)(a2 + 20);
  *v7 = v5;
  sub_264F9A0(v23);
  v14 = v23;
  v15 = *(__int64 **)(v23 + 96);
  v16 = *(__int64 **)(v23 + 104);
  if ( v15 != v16 )
  {
    do
    {
      v17 = *v15++;
      sub_264FAB0(v17, a2);
    }
    while ( v16 != v15 );
    v14 = v23;
  }
  sub_2640E50(&v24, (_QWORD *)(v14 + 72), v13);
  v18 = v24;
  v19 = v25;
  if ( v24 != (__int64 *)v25 )
  {
    do
    {
      v20 = *(_QWORD *)(*v18 + 8);
      if ( *(_QWORD *)*v18 || v20 )
        sub_264FAB0(v20, a2);
      v18 += 2;
    }
    while ( (__int64 *)v19 != v18 );
    v21 = v25;
    v19 = (unsigned __int64)v24;
    if ( (__int64 *)v25 != v24 )
    {
      do
      {
        v22 = *(volatile signed __int32 **)(v19 + 8);
        if ( v22 )
          sub_A191D0(v22);
        v19 += 16LL;
      }
      while ( v21 != v19 );
      v19 = (unsigned __int64)v24;
    }
  }
  if ( v19 )
    j_j___libc_free_0(v19);
}
