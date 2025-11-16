// Function: sub_36FE8C0
// Address: 0x36fe8c0
//
void __fastcall sub_36FE8C0(__int64 a1)
{
  _QWORD *v2; // r13
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // r8
  __int64 v5; // r14
  __int64 v6; // r14
  __int64 v7; // r12
  _QWORD *v8; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 *v11; // r14
  unsigned __int64 *v12; // r12
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 *v17; // r13
  unsigned __int64 *v18; // r12
  unsigned __int64 v19; // rdi
  unsigned __int64 *v20; // r13
  unsigned __int64 *v21; // r12
  unsigned __int64 v22; // rdi
  int v23[9]; // [rsp+Ch] [rbp-24h] BYREF

  *(_QWORD *)a1 = &unk_4A3C580;
  v23[0] = *(_DWORD *)(a1 + 72);
  sub_C83820(v23);
  v2 = *(_QWORD **)(a1 + 240);
  if ( v2 )
  {
    v3 = v2[18];
    if ( (_QWORD *)v3 != v2 + 20 )
      j_j___libc_free_0(v3);
    v4 = v2[15];
    if ( *((_DWORD *)v2 + 33) )
    {
      v5 = *((unsigned int *)v2 + 32);
      if ( (_DWORD)v5 )
      {
        v6 = 8 * v5;
        v7 = 0;
        do
        {
          v8 = *(_QWORD **)(v4 + v7);
          if ( v8 && v8 != (_QWORD *)-8LL )
          {
            sub_C7D6A0((__int64)v8, *v8 + 17LL, 8);
            v4 = v2[15];
          }
          v7 += 8;
        }
        while ( v6 != v7 );
      }
    }
    _libc_free(v4);
    v9 = v2[9];
    if ( v9 )
      j_j___libc_free_0(v9);
    v10 = v2[4];
    if ( (_QWORD *)v10 != v2 + 6 )
      j_j___libc_free_0(v10);
    v11 = (unsigned __int64 *)v2[2];
    v12 = (unsigned __int64 *)v2[1];
    if ( v11 != v12 )
    {
      do
      {
        v13 = v12[5];
        if ( v13 )
          j_j___libc_free_0(v13);
        if ( (unsigned __int64 *)*v12 != v12 + 2 )
          j_j___libc_free_0(*v12);
        v12 += 10;
      }
      while ( v11 != v12 );
      v12 = (unsigned __int64 *)v2[1];
    }
    if ( v12 )
      j_j___libc_free_0((unsigned __int64)v12);
    if ( *v2 )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v2 + 8LL))(*v2);
    j_j___libc_free_0((unsigned __int64)v2);
  }
  v14 = *(_QWORD *)(a1 + 216);
  if ( v14 )
    j_j___libc_free_0(v14);
  v15 = *(_QWORD *)(a1 + 144);
  if ( v15 )
    j_j___libc_free_0(v15);
  v16 = *(_QWORD *)(a1 + 104);
  if ( v16 != a1 + 120 )
    j_j___libc_free_0(v16);
  v17 = *(unsigned __int64 **)(a1 + 88);
  v18 = *(unsigned __int64 **)(a1 + 80);
  if ( v17 != v18 )
  {
    do
    {
      v19 = v18[5];
      if ( v19 )
        j_j___libc_free_0(v19);
      if ( (unsigned __int64 *)*v18 != v18 + 2 )
        j_j___libc_free_0(*v18);
      v18 += 10;
    }
    while ( v17 != v18 );
    v18 = *(unsigned __int64 **)(a1 + 80);
  }
  if ( v18 )
    j_j___libc_free_0((unsigned __int64)v18);
  v20 = *(unsigned __int64 **)(a1 + 56);
  v21 = *(unsigned __int64 **)(a1 + 48);
  *(_QWORD *)a1 = &unk_4A399F0;
  if ( v20 != v21 )
  {
    do
    {
      if ( *v21 )
        j_j___libc_free_0(*v21);
      v21 += 3;
    }
    while ( v20 != v21 );
    v21 = *(unsigned __int64 **)(a1 + 48);
  }
  if ( v21 )
    j_j___libc_free_0((unsigned __int64)v21);
  v22 = *(_QWORD *)(a1 + 24);
  if ( v22 )
    j_j___libc_free_0(v22);
}
