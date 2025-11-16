// Function: sub_145E660
// Address: 0x145e660
//
__int64 __fastcall sub_145E660(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // rax
  _QWORD *v8; // rax
  unsigned __int64 v9; // r15
  _QWORD *v10; // rdx
  __int64 v11; // r12
  __int64 v12; // rax
  _QWORD *v13; // rbx
  _QWORD *v14; // r14
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned int v18; // [rsp+8h] [rbp-48h]
  __int64 v19; // [rsp+10h] [rbp-40h]
  _QWORD *v20; // [rsp+18h] [rbp-38h]

  v3 = a2;
  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation");
  v4 = ((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
      | (*(unsigned int *)(a1 + 12) + 2LL)
      | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4;
  v5 = ((v4
       | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | v4
     | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v6 = (v5 | (v5 >> 16) | HIDWORD(v5)) + 1;
  if ( v6 >= a2 )
    v3 = v6;
  v7 = 0xFFFFFFFFLL;
  if ( v3 <= 0xFFFFFFFF )
    v7 = v3;
  v18 = v7;
  v19 = malloc(24 * v7);
  if ( !v19 )
    sub_16BD1C0("Allocation failed");
  v8 = *(_QWORD **)a1;
  v9 = *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v9 )
  {
    v10 = (_QWORD *)v19;
    do
    {
      if ( v10 )
      {
        *v10 = *v8;
        v10[1] = v8[1];
        v10[2] = v8[2];
        v8[2] = 0;
      }
      v8 += 3;
      v10 += 3;
    }
    while ( (_QWORD *)v9 != v8 );
    v20 = *(_QWORD **)a1;
    v9 = *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v9 )
    {
      do
      {
        v11 = *(_QWORD *)(v9 - 8);
        v9 -= 24LL;
        if ( v11 )
        {
          *(_QWORD *)v11 = &unk_49EC708;
          v12 = *(unsigned int *)(v11 + 208);
          if ( (_DWORD)v12 )
          {
            v13 = *(_QWORD **)(v11 + 192);
            v14 = &v13[7 * v12];
            do
            {
              if ( *v13 != -16 && *v13 != -8 )
              {
                v15 = v13[1];
                if ( (_QWORD *)v15 != v13 + 3 )
                  _libc_free(v15);
              }
              v13 += 7;
            }
            while ( v14 != v13 );
          }
          j___libc_free_0(*(_QWORD *)(v11 + 192));
          v16 = *(_QWORD *)(v11 + 40);
          if ( v16 != v11 + 56 )
            _libc_free(v16);
          j_j___libc_free_0(v11, 216);
        }
      }
      while ( (_QWORD *)v9 != v20 );
      v9 = *(_QWORD *)a1;
    }
  }
  if ( v9 != a1 + 16 )
    _libc_free(v9);
  *(_QWORD *)a1 = v19;
  *(_DWORD *)(a1 + 12) = v18;
  return v18;
}
