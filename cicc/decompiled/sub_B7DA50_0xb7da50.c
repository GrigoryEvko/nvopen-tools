// Function: sub_B7DA50
// Address: 0xb7da50
//
__int64 *__fastcall sub_B7DA50(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 v5; // r14
  __int64 (__fastcall *v6)(__int64, __int64); // rax
  _QWORD *v7; // rbx
  _BYTE *v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rax
  size_t v14; // rdx
  _QWORD *v15; // [rsp+10h] [rbp-90h] BYREF
  size_t n; // [rsp+18h] [rbp-88h]
  _QWORD src[2]; // [rsp+20h] [rbp-80h] BYREF
  _QWORD v18[14]; // [rsp+30h] [rbp-70h] BYREF

  if ( !(*(unsigned __int8 (__fastcall **)(__int64, void *))(*(_QWORD *)*a2 + 48LL))(*a2, &unk_4F84053) )
  {
    v12 = *a2;
    *a2 = 0;
    *a1 = v12 | 1;
    return a1;
  }
  v5 = *a2;
  *a2 = 0;
  v6 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v5 + 24LL);
  if ( v6 == sub_9C3610 )
  {
    v18[6] = &v15;
    v15 = src;
    v18[5] = 0x100000000LL;
    n = 0;
    LOBYTE(src[0]) = 0;
    v18[0] = &unk_49DD210;
    memset(&v18[1], 0, 32);
    sub_CB5980(v18, 0, 0, 0);
    (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v5 + 16LL))(v5, v18);
    v18[0] = &unk_49DD210;
    sub_CB5840(v18);
  }
  else
  {
    v6((__int64)&v15, v5);
  }
  v7 = (_QWORD *)*a3;
  v8 = *(_BYTE **)(*a3 + 8);
  if ( v15 == src )
  {
    v14 = n;
    if ( n )
    {
      if ( n == 1 )
        *v8 = src[0];
      else
        memcpy(v8, src, n);
      v14 = n;
      v8 = (_BYTE *)v7[1];
    }
    v7[2] = v14;
    v8[v14] = 0;
    v8 = v15;
    goto LABEL_8;
  }
  if ( v8 == (_BYTE *)(v7 + 3) )
  {
    v7[1] = v15;
    v7[2] = n;
    v7[3] = src[0];
    goto LABEL_20;
  }
  v7[1] = v15;
  v9 = v7[3];
  v7[2] = n;
  v7[3] = src[0];
  if ( !v8 )
  {
LABEL_20:
    v15 = src;
    v8 = src;
    goto LABEL_8;
  }
  v15 = v8;
  src[0] = v9;
LABEL_8:
  n = 0;
  *v8 = 0;
  if ( v15 != src )
    j_j___libc_free_0(v15, src[0] + 1LL);
  v10 = *a3;
  *(_DWORD *)(v10 + 40) = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 32LL))(v5);
  *(_QWORD *)(v10 + 48) = v11;
  *a1 = 1;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v5 + 8LL))(v5);
  return a1;
}
