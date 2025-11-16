// Function: sub_EA3000
// Address: 0xea3000
//
_QWORD *__fastcall sub_EA3000(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  _QWORD *v8; // r8
  __int64 v9; // rax
  __int64 *v10; // rdi
  unsigned __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // rbx
  __int64 v16; // rdi
  __int64 *v17; // rdi
  __int64 v23; // [rsp+20h] [rbp-80h]
  _QWORD *v24; // [rsp+28h] [rbp-78h]
  _QWORD v25[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v26; // [rsp+40h] [rbp-60h] BYREF
  __int64 v27; // [rsp+48h] [rbp-58h]
  __int16 v28; // [rsp+60h] [rbp-40h]

  v6 = *(_QWORD *)(a1 + 16);
  v23 = v6 + 112LL * *(unsigned int *)(a1 + 24);
  if ( v6 != v23 )
  {
    v7 = *(_QWORD *)(a1 + 16);
    v8 = v25;
    do
    {
      v28 = 261;
      v9 = *(_QWORD *)(v7 + 8);
      v10 = *(__int64 **)(a1 + 248);
      v24 = v8;
      v7 += 112;
      v26 = v9;
      v27 = *(_QWORD *)(v7 - 96);
      v11 = *(_QWORD *)(v7 - 112);
      v12 = *(_QWORD *)(v7 - 16);
      v13 = *(_QWORD *)(v7 - 8);
      *(_BYTE *)(a1 + 32) = 1;
      v25[0] = v12;
      v25[1] = v13;
      sub_C91CB0(v10, v11, 0, (__int64)&v26, (__int64)v8, 1, 0, 0, 1u);
      sub_EA2AE0((_QWORD *)a1);
      v8 = v24;
    }
    while ( v23 != v7 );
    v14 = *(_QWORD *)(a1 + 16);
    v15 = v14 + 112LL * *(unsigned int *)(a1 + 24);
    while ( v14 != v15 )
    {
      while ( 1 )
      {
        v15 -= 112;
        v16 = *(_QWORD *)(v15 + 8);
        if ( v16 == v15 + 32 )
          break;
        _libc_free(v16, v11);
        if ( v14 == v15 )
          goto LABEL_8;
      }
    }
  }
LABEL_8:
  *(_DWORD *)(a1 + 24) = 0;
  v17 = *(__int64 **)(a1 + 248);
  v26 = a4;
  v27 = a5;
  sub_C91CB0(v17, a2, 3, a3, (__int64)&v26, 1, 0, 0, 1u);
  return sub_EA2AE0((_QWORD *)a1);
}
