// Function: sub_E9B370
// Address: 0xe9b370
//
__int64 __fastcall sub_E9B370(__int64 *a1, __int64 a2, _QWORD *a3)
{
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 result; // rax
  int v9; // eax
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 (*v12)(); // rax
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 *v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // r14
  __int64 v18; // r15
  __int64 v19; // rdi
  __int64 v20; // r13
  __int64 v21; // rsi
  __int64 v22; // rdi
  __int64 v23; // r14
  __int64 v24; // r13
  __int64 v25; // rdi
  __int64 v26; // rdi
  __int64 v27[4]; // [rsp+10h] [rbp-60h] BYREF
  char v28; // [rsp+30h] [rbp-40h]
  char v29; // [rsp+31h] [rbp-3Fh]

  v6 = a1[1];
  v7 = *(_QWORD *)(v6 + 152);
  if ( *(_DWORD *)(v7 + 336) != 4 || (v9 = *(_DWORD *)(v7 + 344), v9 == 6) || !v9 )
  {
    v29 = 1;
    v27[0] = (__int64)".seh_* directives are not supported on this target";
    v28 = 3;
    return sub_E66880(v6, a3, (__int64)v27);
  }
  v10 = a1[13];
  if ( v10 && !*(_QWORD *)(v10 + 8) )
  {
    v29 = 1;
    v27[0] = (__int64)"Starting a function before ending the previous one!";
    v28 = 3;
    sub_E66880(v6, a3, (__int64)v27);
  }
  v11 = 1;
  v12 = *(__int64 (**)())(*a1 + 88);
  if ( v12 != sub_E97650 )
    v11 = ((__int64 (__fastcall *)(__int64 *, _QWORD *))v12)(a1, a3);
  a1[14] = (a1[11] - a1[10]) >> 3;
  v13 = sub_22077B0(184);
  v14 = v13;
  if ( v13 )
  {
    *(_QWORD *)v13 = v11;
    *(_QWORD *)(v13 + 8) = 0;
    *(_QWORD *)(v13 + 16) = 0;
    *(_QWORD *)(v13 + 24) = 0;
    *(_QWORD *)(v13 + 32) = a2;
    *(_QWORD *)(v13 + 40) = 0;
    *(_QWORD *)(v13 + 48) = 0;
    *(_QWORD *)(v13 + 56) = 0;
    *(_QWORD *)(v13 + 64) = 0;
    *(_QWORD *)(v13 + 72) = 0xFFFFFFFF00000000LL;
    *(_QWORD *)(v13 + 80) = 0;
    *(_QWORD *)(v13 + 88) = 0;
    *(_QWORD *)(v13 + 96) = 0;
    *(_QWORD *)(v13 + 104) = 0;
    *(_QWORD *)(v13 + 112) = 0;
    *(_QWORD *)(v13 + 120) = 0;
    *(_QWORD *)(v13 + 128) = 0;
    *(_DWORD *)(v13 + 136) = 0;
    *(_QWORD *)(v13 + 144) = v13 + 160;
    *(_QWORD *)(v13 + 152) = 0;
    *(_QWORD *)(v13 + 160) = 0;
    *(_QWORD *)(v13 + 168) = 0;
    *(_QWORD *)(v13 + 176) = 0;
  }
  v27[0] = v13;
  v15 = (__int64 *)a1[11];
  if ( v15 == (__int64 *)a1[12] )
  {
    sub_E9B0A0(a1 + 10, (__int64)v15, v27);
    v14 = v27[0];
LABEL_17:
    if ( v14 )
    {
      v17 = *(_QWORD *)(v14 + 168);
      v18 = *(_QWORD *)(v14 + 160);
      if ( v17 != v18 )
      {
        do
        {
          v19 = *(_QWORD *)(v18 + 64);
          v20 = v18 + 80;
          if ( v19 != v18 + 80 )
            _libc_free(v19, v15);
          v21 = *(unsigned int *)(v18 + 56);
          v22 = *(_QWORD *)(v18 + 40);
          v18 += 80;
          v15 = (__int64 *)(16 * v21);
          sub_C7D6A0(v22, (__int64)v15, 8);
        }
        while ( v17 != v20 );
        v18 = *(_QWORD *)(v14 + 160);
      }
      if ( v18 )
      {
        v15 = (__int64 *)(*(_QWORD *)(v14 + 176) - v18);
        j_j___libc_free_0(v18, v15);
      }
      v23 = *(_QWORD *)(v14 + 144);
      v24 = v23 + 48LL * *(unsigned int *)(v14 + 152);
      if ( v23 != v24 )
      {
        do
        {
          v25 = *(_QWORD *)(v24 - 40);
          v24 -= 48;
          if ( v25 )
          {
            v15 = (__int64 *)(*(_QWORD *)(v24 + 24) - v25);
            j_j___libc_free_0(v25, v15);
          }
        }
        while ( v23 != v24 );
        v24 = *(_QWORD *)(v14 + 144);
      }
      if ( v14 + 160 != v24 )
        _libc_free(v24, v15);
      sub_C7D6A0(*(_QWORD *)(v14 + 120), 16LL * *(unsigned int *)(v14 + 136), 8);
      v26 = *(_QWORD *)(v14 + 88);
      if ( v26 )
        j_j___libc_free_0(v26, *(_QWORD *)(v14 + 104) - v26);
      j_j___libc_free_0(v14, 184);
    }
    goto LABEL_15;
  }
  if ( !v15 )
  {
    a1[11] = 8;
    goto LABEL_17;
  }
  *v15 = v13;
  a1[11] += 8;
LABEL_15:
  v16 = a1[36];
  result = *(_QWORD *)(a1[11] - 8);
  a1[13] = result;
  *(_QWORD *)(result + 56) = *(_QWORD *)(v16 + 8);
  return result;
}
