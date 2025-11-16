// Function: sub_2B15EA0
// Address: 0x2b15ea0
//
__int64 __fastcall sub_2B15EA0(__int64 a1, unsigned int *a2)
{
  __int64 v2; // r9
  unsigned __int64 v3; // r13
  int v5; // r15d
  _QWORD *v6; // rbx
  __int64 v7; // rax
  int v8; // r14d
  unsigned int v9; // edx
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned __int64 v15; // rdi
  __int64 v16; // rcx
  __int64 v17; // rax
  unsigned int v18; // ebx
  __int64 v19; // [rsp+8h] [rbp-48h]
  unsigned __int64 v20; // [rsp+10h] [rbp-40h] BYREF
  unsigned __int64 v21; // [rsp+18h] [rbp-38h]

  v21 = sub_D35010(
          **(_QWORD **)(a1 + 16),
          *(_QWORD *)(*(_QWORD *)a2 + 8LL),
          **(_QWORD **)(a1 + 16),
          **(_QWORD **)(a1 + 8),
          *(_QWORD *)(a1 + 24),
          *(_QWORD *)(a1 + 32),
          1,
          1);
  v3 = HIDWORD(v21);
  if ( BYTE4(v21) )
  {
    v5 = v21;
    v6 = *(_QWORD **)(a1 + 8);
    v7 = a2[2];
    v8 = **(_QWORD **)a1 + 1;
    v9 = v7;
    if ( a2[3] <= (unsigned int)v7 )
    {
      v11 = sub_C8D7D0((__int64)a2, (__int64)(a2 + 4), 0, 0x10u, &v20, v2);
      v12 = 16LL * a2[2];
      v13 = v12 + v11;
      if ( v12 + v11 )
      {
        *(_DWORD *)v13 = v8;
        *(_DWORD *)(v13 + 4) = v5;
        *(_QWORD *)(v13 + 8) = *v6;
        v12 = 16LL * a2[2];
      }
      v14 = *(_QWORD *)a2;
      v15 = *(_QWORD *)a2 + v12;
      if ( *(_QWORD *)a2 != v15 )
      {
        v16 = v11 + v12;
        v17 = v11;
        do
        {
          if ( v17 )
          {
            *(_DWORD *)v17 = *(_DWORD *)v14;
            *(_DWORD *)(v17 + 4) = *(_DWORD *)(v14 + 4);
            *(_QWORD *)(v17 + 8) = *(_QWORD *)(v14 + 8);
          }
          v17 += 16;
          v14 += 16;
        }
        while ( v16 != v17 );
        v15 = *(_QWORD *)a2;
      }
      v18 = v20;
      if ( a2 + 4 != (unsigned int *)v15 )
      {
        v19 = v11;
        _libc_free(v15);
        v11 = v19;
      }
      ++a2[2];
      *(_QWORD *)a2 = v11;
      a2[3] = v18;
    }
    else
    {
      v10 = *(_QWORD *)a2 + 16 * v7;
      if ( v10 )
      {
        *(_DWORD *)v10 = v8;
        *(_DWORD *)(v10 + 4) = v5;
        *(_QWORD *)(v10 + 8) = *v6;
        v9 = a2[2];
      }
      a2[2] = v9 + 1;
    }
  }
  return (unsigned int)v3;
}
