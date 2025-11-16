// Function: sub_A555E0
// Address: 0xa555e0
//
__int64 __fastcall sub_A555E0(__int64 a1)
{
  __int64 v2; // rsi
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // r13
  __int64 v8; // r15
  __int64 v9; // r15
  __int64 v10; // r12
  __int64 v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rsi
  __int64 result; // rax
  __int64 v18; // r12
  __int64 v19; // rdi
  __int64 (__fastcall *v20)(__int64, __int64); // rax

  v2 = 16LL * *(unsigned int *)(a1 + 672);
  sub_C7D6A0(*(_QWORD *)(a1 + 656), v2, 8);
  v3 = *(_QWORD *)(a1 + 504);
  if ( v3 != a1 + 520 )
    _libc_free(v3, v2);
  v4 = *(_QWORD *)(a1 + 360);
  if ( v4 != a1 + 376 )
    _libc_free(v4, v2);
  v5 = *(unsigned int *)(a1 + 352);
  if ( (_DWORD)v5 )
  {
    v6 = *(_QWORD *)(a1 + 336);
    v7 = v6 + 56 * v5;
    do
    {
      v8 = v6 + 56;
      if ( *(_QWORD *)v6 != -8192 && *(_QWORD *)v6 != -4096 )
      {
        v9 = *(_QWORD *)(v6 + 40);
        v10 = v9 + 32LL * *(unsigned int *)(v6 + 48);
        if ( v9 != v10 )
        {
          do
          {
            v11 = *(_QWORD *)(v10 - 24);
            v10 -= 32;
            if ( v11 )
            {
              v2 = *(_QWORD *)(v10 + 24) - v11;
              j_j___libc_free_0(v11, v2);
            }
          }
          while ( v9 != v10 );
          v10 = *(_QWORD *)(v6 + 40);
        }
        v8 = v6 + 56;
        if ( v10 != v6 + 56 )
          _libc_free(v10, v2);
        v2 = 16LL * *(unsigned int *)(v6 + 32);
        sub_C7D6A0(*(_QWORD *)(v6 + 16), v2, 8);
      }
      v6 = v8;
    }
    while ( v7 != v8 );
    v5 = *(unsigned int *)(a1 + 352);
  }
  v12 = 56 * v5;
  sub_C7D6A0(*(_QWORD *)(a1 + 336), 56 * v5, 8);
  v13 = *(_QWORD *)(a1 + 304);
  if ( v13 != a1 + 320 )
    _libc_free(v13, v12);
  sub_C7D6A0(*(_QWORD *)(a1 + 280), 8LL * *(unsigned int *)(a1 + 296), 8);
  v14 = *(_QWORD *)(a1 + 240);
  if ( v14 )
    j_j___libc_free_0(v14, *(_QWORD *)(a1 + 256) - v14);
  sub_C7D6A0(*(_QWORD *)(a1 + 216), 16LL * *(unsigned int *)(a1 + 232), 8);
  v15 = *(_QWORD *)(a1 + 176);
  if ( v15 )
    j_j___libc_free_0(v15, *(_QWORD *)(a1 + 192) - v15);
  sub_C7D6A0(*(_QWORD *)(a1 + 152), 8LL * *(unsigned int *)(a1 + 168), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 120), 8LL * *(unsigned int *)(a1 + 136), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 88), 8LL * *(unsigned int *)(a1 + 104), 8);
  v16 = 8LL * *(unsigned int *)(a1 + 72);
  result = sub_C7D6A0(*(_QWORD *)(a1 + 56), v16, 8);
  v18 = *(_QWORD *)(a1 + 24);
  if ( v18 )
  {
    v19 = *(_QWORD *)(a1 + 24);
    v20 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v18 + 8LL);
    if ( v20 == sub_A554F0 )
    {
      sub_A552A0(v19, v16);
      return j_j___libc_free_0(v18, 400);
    }
    else
    {
      return ((__int64 (__fastcall *)(__int64))v20)(v19);
    }
  }
  return result;
}
