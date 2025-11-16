// Function: sub_2855140
// Address: 0x2855140
//
void __fastcall sub_2855140(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  int v8; // eax
  _QWORD *v9; // rdi
  __int64 v10; // rax
  _QWORD *v11; // r15
  __int64 v12; // r14
  __int64 v13; // r12
  _QWORD *v14; // rdi
  __int64 v15; // rax
  _QWORD *v16; // rax
  _QWORD *v17; // r12
  unsigned __int64 *v18; // r13
  unsigned __int64 v19; // rdx
  _QWORD *v20; // r13
  __int64 v21; // rax
  int v22; // r13d
  _QWORD *v23; // [rsp+8h] [rbp-48h]
  unsigned __int64 v24[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = *(unsigned int *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v7 )
  {
    v11 = (_QWORD *)(a1 + 16);
    v12 = sub_C8D7D0(a1, a1 + 16, 0, 0x18u, v24, a6);
    v13 = 24LL * *(unsigned int *)(a1 + 8);
    v14 = (_QWORD *)(v13 + v12);
    if ( v13 + v12 )
    {
      v15 = *a2;
      *v14 = 6;
      v14[1] = 0;
      v14[2] = v15;
      if ( v15 != 0 && v15 != -4096 && v15 != -8192 )
        sub_BD73F0((__int64)v14);
      v13 = 24LL * *(unsigned int *)(a1 + 8);
    }
    v16 = *(_QWORD **)a1;
    v17 = (_QWORD *)(*(_QWORD *)a1 + v13);
    if ( *(_QWORD **)a1 != v17 )
    {
      v18 = (unsigned __int64 *)v12;
      do
      {
        if ( v18 )
        {
          *v18 = 6;
          v18[1] = 0;
          v19 = v16[2];
          v18[2] = v19;
          if ( v19 != 0 && v19 != -4096 && v19 != -8192 )
          {
            v23 = v16;
            sub_BD6050(v18, *v16 & 0xFFFFFFFFFFFFFFF8LL);
            v16 = v23;
          }
        }
        v16 += 3;
        v18 += 3;
      }
      while ( v17 != v16 );
      v20 = *(_QWORD **)a1;
      v17 = (_QWORD *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8));
      if ( *(_QWORD **)a1 != v17 )
      {
        do
        {
          v21 = *(v17 - 1);
          v17 -= 3;
          if ( v21 != 0 && v21 != -4096 && v21 != -8192 )
            sub_BD60C0(v17);
        }
        while ( v17 != v20 );
        v17 = *(_QWORD **)a1;
      }
    }
    v22 = v24[0];
    if ( v11 != v17 )
      _libc_free((unsigned __int64)v17);
    *(_QWORD *)a1 = v12;
    *(_DWORD *)(a1 + 12) = v22;
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    v8 = *(_DWORD *)(a1 + 8);
    v9 = (_QWORD *)(*(_QWORD *)a1 + 24 * v7);
    if ( v9 )
    {
      v10 = *a2;
      *v9 = 6;
      v9[1] = 0;
      v9[2] = v10;
      if ( v10 != 0 && v10 != -4096 && v10 != -8192 )
        sub_BD73F0((__int64)v9);
      v8 = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = v8 + 1;
  }
}
