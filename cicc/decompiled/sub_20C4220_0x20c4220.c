// Function: sub_20C4220
// Address: 0x20c4220
//
__int64 __fastcall sub_20C4220(__int64 a1, __int64 *a2, unsigned int a3)
{
  __int64 v5; // rax
  int v6; // r12d
  unsigned int v7; // r12d
  __int64 v8; // rax
  size_t v9; // rdx
  void *v10; // r13
  __int64 v11; // rax
  size_t v12; // rdx
  __int64 v13; // r12
  char v14; // r15
  _QWORD *v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rcx
  int v18; // r8d
  int v19; // r9d
  unsigned int v20; // edi
  unsigned int v21; // eax
  __int64 v22; // rdx
  _QWORD *v23; // rcx
  __int64 v24; // rsi
  __int64 v25; // rcx
  __int64 v27; // rax
  size_t n; // [rsp+0h] [rbp-60h]
  size_t na; // [rsp+0h] [rbp-60h]
  unsigned int v30; // [rsp+Ch] [rbp-54h] BYREF
  unsigned __int64 v31[2]; // [rsp+10h] [rbp-50h] BYREF
  int v32; // [rsp+20h] [rbp-40h]

  v5 = a2[4];
  v30 = a3;
  v6 = *(_DWORD *)(v5 + 16);
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = v6;
  v7 = (unsigned int)(v6 + 63) >> 6;
  v8 = malloc(8LL * v7);
  v9 = 8LL * v7;
  v10 = (void *)v8;
  if ( !v8 )
  {
    if ( 8LL * v7 || (v27 = malloc(1u), v9 = 0, !v27) )
    {
      na = v9;
      sub_16BD1C0("Allocation failed", 1u);
      v9 = na;
    }
    else
    {
      v10 = (void *)v27;
    }
  }
  *(_QWORD *)a1 = v10;
  *(_QWORD *)(a1 + 8) = v7;
  if ( v7 )
    memset(v10, 0, v9);
  v11 = sub_20C3470(a2[9] + 56, &v30);
  n = v12;
  v13 = v11;
  if ( v12 != v11 )
  {
    v14 = 1;
    do
    {
      v15 = *(_QWORD **)(v13 + 48);
      if ( v15 )
      {
        sub_1F4AD00((__int64)v31, a2[4], a2[1], v15);
        if ( v14 )
        {
          sub_20C1FD0(a1, (__int64)v31, v16, v17, v18, v19);
        }
        else
        {
          v20 = (unsigned int)(*(_DWORD *)(a1 + 16) + 63) >> 6;
          v21 = (unsigned int)(v32 + 63) >> 6;
          if ( v21 > v20 )
            v21 = (unsigned int)(*(_DWORD *)(a1 + 16) + 63) >> 6;
          v22 = 0;
          if ( v21 )
          {
            do
            {
              v23 = (_QWORD *)(v22 + *(_QWORD *)a1);
              v24 = *(_QWORD *)(v31[0] + v22);
              v22 += 8;
              *v23 &= v24;
            }
            while ( v22 != 8LL * v21 );
          }
          for ( ; v20 != v21; *(_QWORD *)(*(_QWORD *)a1 + 8 * v25) = 0 )
            v25 = v21++;
        }
        v14 = 0;
        _libc_free(v31[0]);
      }
      v13 = sub_220EEE0(v13);
    }
    while ( n != v13 );
  }
  return a1;
}
