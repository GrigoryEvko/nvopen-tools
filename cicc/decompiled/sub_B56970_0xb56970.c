// Function: sub_B56970
// Address: 0xb56970
//
void __fastcall sub_B56970(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 *v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rcx
  int v19; // eax
  __int64 v20; // rdi
  __m128i *v21; // r15
  __int64 v22; // rdi
  int v23; // edx
  int v24; // [rsp-70h] [rbp-70h]
  int v25; // [rsp-60h] [rbp-60h] BYREF
  _QWORD v26[11]; // [rsp-58h] [rbp-58h] BYREF

  if ( *(char *)(a1 + 7) < 0 )
  {
    v4 = sub_BD2BC0(a1);
    v6 = v4 + v5;
    if ( *(char *)(a1 + 7) >= 0 )
      v7 = v6 >> 4;
    else
      LODWORD(v7) = (v6 - sub_BD2BC0(a1)) >> 4;
    v8 = 0;
    v9 = 16LL * (unsigned int)v7;
    if ( (_DWORD)v7 )
    {
      do
      {
        v10 = 0;
        if ( *(char *)(a1 + 7) < 0 )
          v10 = sub_BD2BC0(a1);
        v11 = (__int64 *)(v8 + v10);
        v12 = *((unsigned int *)v11 + 2);
        v13 = *v11;
        v14 = *((unsigned int *)v11 + 3);
        v15 = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
        v26[2] = v13;
        v16 = 32 * v14 - 32 * v12;
        v17 = 32 * v12 - v15;
        v18 = *(unsigned int *)(a2 + 8);
        v26[1] = v16 >> 5;
        v19 = v18;
        v26[0] = a1 + v17;
        if ( *(_DWORD *)(a2 + 12) <= (unsigned int)v18 )
        {
          v21 = (__m128i *)sub_C8D7D0(a2, a2 + 16, 0, 56, &v25);
          v22 = (__int64)&v21->m128i_i64[7 * *(unsigned int *)(a2 + 8)];
          if ( v22 )
            sub_B56460(v22, (__int64)v26);
          sub_B56820(a2, v21);
          v23 = v25;
          if ( a2 + 16 != *(_QWORD *)a2 )
          {
            v24 = v25;
            _libc_free(*(_QWORD *)a2, v21);
            v23 = v24;
          }
          ++*(_DWORD *)(a2 + 8);
          *(_QWORD *)a2 = v21;
          *(_DWORD *)(a2 + 12) = v23;
        }
        else
        {
          v20 = *(_QWORD *)a2 + 56 * v18;
          if ( v20 )
          {
            sub_B56460(v20, (__int64)v26);
            v19 = *(_DWORD *)(a2 + 8);
          }
          *(_DWORD *)(a2 + 8) = v19 + 1;
        }
        v8 += 16;
      }
      while ( v8 != v9 );
    }
  }
}
