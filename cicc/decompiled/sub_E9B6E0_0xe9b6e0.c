// Function: sub_E9B6E0
// Address: 0xe9b6e0
//
__int64 __fastcall sub_E9B6E0(__int64 *a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 (*v6)(); // rax
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 *v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // r14
  __int64 v13; // r15
  __int64 v14; // rdi
  __int64 v15; // rbx
  __int64 v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // rbx
  __int64 v19; // r14
  __int64 v20; // rdi
  __int64 v21; // rdi
  __int64 v22[7]; // [rsp+18h] [rbp-38h] BYREF

  result = sub_E99590((__int64)a1, a2);
  if ( result )
  {
    v4 = result;
    v5 = 1;
    v6 = *(__int64 (**)())(*a1 + 88);
    if ( v6 != sub_E97650 )
      v5 = ((__int64 (__fastcall *)(__int64 *))v6)(a1);
    v7 = *(_QWORD *)(v4 + 32);
    v8 = sub_22077B0(184);
    v9 = v8;
    if ( v8 )
    {
      *(_QWORD *)v8 = v5;
      *(_QWORD *)(v8 + 8) = 0;
      *(_QWORD *)(v8 + 16) = 0;
      *(_QWORD *)(v8 + 24) = 0;
      *(_QWORD *)(v8 + 32) = v7;
      *(_QWORD *)(v8 + 40) = 0;
      *(_QWORD *)(v8 + 48) = 0;
      *(_QWORD *)(v8 + 56) = 0;
      *(_QWORD *)(v8 + 64) = 0;
      *(_QWORD *)(v8 + 72) = 0xFFFFFFFF00000000LL;
      *(_QWORD *)(v8 + 80) = v4;
      *(_QWORD *)(v8 + 88) = 0;
      *(_QWORD *)(v8 + 96) = 0;
      *(_QWORD *)(v8 + 104) = 0;
      *(_QWORD *)(v8 + 112) = 0;
      *(_QWORD *)(v8 + 120) = 0;
      *(_QWORD *)(v8 + 128) = 0;
      *(_DWORD *)(v8 + 136) = 0;
      *(_QWORD *)(v8 + 144) = v8 + 160;
      *(_QWORD *)(v8 + 152) = 0;
      *(_QWORD *)(v8 + 160) = 0;
      *(_QWORD *)(v8 + 168) = 0;
      *(_QWORD *)(v8 + 176) = 0;
    }
    v22[0] = v8;
    v10 = (__int64 *)a1[11];
    if ( v10 == (__int64 *)a1[12] )
    {
      sub_E9B0A0(a1 + 10, (__int64)v10, v22);
      v9 = v22[0];
    }
    else
    {
      if ( v10 )
      {
        *v10 = v8;
        a1[11] += 8;
LABEL_9:
        v11 = a1[36];
        result = *(_QWORD *)(a1[11] - 8);
        a1[13] = result;
        *(_QWORD *)(result + 56) = *(_QWORD *)(v11 + 8);
        return result;
      }
      a1[11] = 8;
    }
    if ( v9 )
    {
      v12 = *(_QWORD *)(v9 + 168);
      v13 = *(_QWORD *)(v9 + 160);
      if ( v12 != v13 )
      {
        do
        {
          v14 = *(_QWORD *)(v13 + 64);
          v15 = v13 + 80;
          if ( v14 != v13 + 80 )
            _libc_free(v14, v10);
          v16 = *(unsigned int *)(v13 + 56);
          v17 = *(_QWORD *)(v13 + 40);
          v13 += 80;
          v10 = (__int64 *)(16 * v16);
          sub_C7D6A0(v17, (__int64)v10, 8);
        }
        while ( v12 != v15 );
        v13 = *(_QWORD *)(v9 + 160);
      }
      if ( v13 )
      {
        v10 = (__int64 *)(*(_QWORD *)(v9 + 176) - v13);
        j_j___libc_free_0(v13, v10);
      }
      v18 = *(_QWORD *)(v9 + 144);
      v19 = v18 + 48LL * *(unsigned int *)(v9 + 152);
      if ( v18 != v19 )
      {
        do
        {
          v20 = *(_QWORD *)(v19 - 40);
          v19 -= 48;
          if ( v20 )
          {
            v10 = (__int64 *)(*(_QWORD *)(v19 + 24) - v20);
            j_j___libc_free_0(v20, v10);
          }
        }
        while ( v18 != v19 );
        v19 = *(_QWORD *)(v9 + 144);
      }
      if ( v19 != v9 + 160 )
        _libc_free(v19, v10);
      sub_C7D6A0(*(_QWORD *)(v9 + 120), 16LL * *(unsigned int *)(v9 + 136), 8);
      v21 = *(_QWORD *)(v9 + 88);
      if ( v21 )
        j_j___libc_free_0(v21, *(_QWORD *)(v9 + 104) - v21);
      j_j___libc_free_0(v9, 184);
    }
    goto LABEL_9;
  }
  return result;
}
