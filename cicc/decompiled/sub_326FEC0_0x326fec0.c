// Function: sub_326FEC0
// Address: 0x326fec0
//
__int64 __fastcall sub_326FEC0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int16 v7; // r12
  unsigned int v8; // r13d
  unsigned int v9; // r14d
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rdi
  int v14; // eax
  __int64 v15; // rdi
  __int64 v16; // r13
  __int64 v17; // rcx
  __int64 v18; // rdi
  unsigned int v19; // eax
  unsigned __int64 v20; // r15
  __int64 v21; // rax
  _DWORD *v22; // rdx
  __int64 v23; // rbx
  int v24; // ebx
  unsigned __int64 v26[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = *(_QWORD *)(*(_QWORD *)a2 + 96LL);
  v7 = *(_WORD *)(*(_QWORD *)a2 + 32LL);
  v8 = *(_DWORD *)(v6 + 32);
  if ( v8 <= 0x40 )
  {
    v11 = *(_QWORD *)(v6 + 24);
    v9 = 0;
    if ( !v11 || (v7 & 8) != 0 || (v11 & (v11 - 1)) != 0 )
      return v9;
  }
  else
  {
    v9 = 0;
    if ( v8 == (unsigned int)sub_C444A0(v6 + 24) || (v7 & 8) != 0 || (unsigned int)sub_C44630(v6 + 24) != 1 )
      return v9;
  }
  v12 = *a1;
  v13 = *(unsigned int *)(*a1 + 8);
  v14 = v13;
  if ( *(_DWORD *)(*a1 + 12) <= (unsigned int)v13 )
  {
    v16 = sub_C8D7D0(*a1, v12 + 16, 0, 0x10u, v26, a6);
    v17 = 16LL * *(unsigned int *)(v12 + 8);
    v18 = v17 + v16;
    if ( v17 + v16 )
    {
      v19 = *(_DWORD *)(v6 + 32);
      *(_DWORD *)(v18 + 8) = v19;
      if ( v19 > 0x40 )
        sub_C43780(v18, (const void **)(v6 + 24));
      else
        *(_QWORD *)v18 = *(_QWORD *)(v6 + 24);
      v17 = 16LL * *(unsigned int *)(v12 + 8);
    }
    v20 = *(_QWORD *)v12;
    v21 = v16;
    v22 = (_DWORD *)(*(_QWORD *)v12 + 8LL);
    if ( v17 )
    {
      do
      {
        if ( v21 )
        {
          *(_DWORD *)(v21 + 8) = *v22;
          *(_QWORD *)v21 = *((_QWORD *)v22 - 1);
          *v22 = 0;
        }
        v21 += 16;
        v22 += 4;
      }
      while ( v21 != v16 + v17 );
      v20 = *(_QWORD *)v12;
      v23 = *(_QWORD *)v12 + 16LL * *(unsigned int *)(v12 + 8);
      if ( *(_QWORD *)v12 != v23 )
      {
        do
        {
          v23 -= 16;
          if ( *(_DWORD *)(v23 + 8) > 0x40u && *(_QWORD *)v23 )
            j_j___libc_free_0_0(*(_QWORD *)v23);
        }
        while ( v20 != v23 );
        v20 = *(_QWORD *)v12;
      }
    }
    v24 = v26[0];
    if ( v12 + 16 != v20 )
      _libc_free(v20);
    ++*(_DWORD *)(v12 + 8);
    v9 = 1;
    *(_QWORD *)v12 = v16;
    *(_DWORD *)(v12 + 12) = v24;
  }
  else
  {
    v15 = *(_QWORD *)v12 + 16 * v13;
    if ( v15 )
    {
      *(_DWORD *)(v15 + 8) = v8;
      if ( v8 > 0x40 )
        sub_C43780(v15, (const void **)(v6 + 24));
      else
        *(_QWORD *)v15 = *(_QWORD *)(v6 + 24);
      v14 = *(_DWORD *)(v12 + 8);
    }
    v9 = 1;
    *(_DWORD *)(v12 + 8) = v14 + 1;
  }
  return v9;
}
