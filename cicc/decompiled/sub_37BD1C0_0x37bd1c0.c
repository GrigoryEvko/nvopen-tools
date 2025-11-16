// Function: sub_37BD1C0
// Address: 0x37bd1c0
//
void __fastcall sub_37BD1C0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // r13
  __int64 v8; // rbx
  __int64 v9; // rdx
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rcx
  char *v12; // r12
  unsigned __int64 v13; // rsi
  int v14; // eax
  _QWORD *v15; // rdx
  unsigned __int64 v16; // r14
  _QWORD *v17; // rax
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r12
  _QWORD *v21; // rax
  unsigned __int64 v22; // rdx
  char *v23; // r12
  __int64 v24; // [rsp-58h] [rbp-58h]
  _QWORD v25[8]; // [rsp-40h] [rbp-40h] BYREF

  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)a1 = a1 + 16;
  if ( a2 )
  {
    v7 = (int)a3;
    v8 = 0;
    sub_37BD0D0(a1, (int)a2, a3, a4, a5, a6);
    v24 = a2;
    do
    {
      v17 = (_QWORD *)sub_22077B0(0x10u);
      v16 = (unsigned __int64)v17;
      if ( v17 )
      {
        v17[1] = 0;
        *v17 = v17 + 2;
        v20 = unk_5051170;
        if ( v7 )
        {
          sub_C8D5F0((__int64)v17, v17 + 2, v7, 8u, v18, v19);
          v21 = *(_QWORD **)v16;
          v22 = v7;
          do
          {
            if ( v21 )
              *v21 = v20;
            ++v21;
            --v22;
          }
          while ( v22 );
          *(_DWORD *)(v16 + 8) = v7;
        }
      }
      v9 = *(unsigned int *)(a1 + 8);
      v10 = *(unsigned int *)(a1 + 12);
      v25[0] = v16;
      v11 = *(_QWORD *)a1;
      v12 = (char *)v25;
      v13 = v9 + 1;
      v14 = v9;
      if ( v9 + 1 > v10 )
      {
        if ( v11 > (unsigned __int64)v25 || (unsigned __int64)v25 >= v11 + 8 * v9 )
        {
          sub_37BD0D0(a1, v13, v9, v11, v18, v19);
          v9 = *(unsigned int *)(a1 + 8);
          v11 = *(_QWORD *)a1;
          v14 = *(_DWORD *)(a1 + 8);
        }
        else
        {
          v23 = (char *)v25 - v11;
          sub_37BD0D0(a1, v13, v9, v11, v18, v19);
          v11 = *(_QWORD *)a1;
          v9 = *(unsigned int *)(a1 + 8);
          v12 = &v23[*(_QWORD *)a1];
          v14 = *(_DWORD *)(a1 + 8);
        }
      }
      v15 = (_QWORD *)(v11 + 8 * v9);
      if ( v15 )
      {
        *v15 = *(_QWORD *)v12;
        *(_QWORD *)v12 = 0;
        v16 = v25[0];
        v14 = *(_DWORD *)(a1 + 8);
      }
      *(_DWORD *)(a1 + 8) = v14 + 1;
      if ( v16 )
      {
        if ( *(_QWORD *)v16 != v16 + 16 )
          _libc_free(*(_QWORD *)v16);
        j_j___libc_free_0(v16);
      }
      ++v8;
    }
    while ( v24 != v8 );
  }
}
