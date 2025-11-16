// Function: sub_E972E0
// Address: 0xe972e0
//
__int64 __fastcall sub_E972E0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rdi
  void *v7; // rsi
  __int64 result; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  void *v11; // r15
  __int64 v12; // rdi
  __int64 v13; // rbx
  size_t v14; // r12
  size_t v15; // r12
  _BYTE *v16; // rdi
  _QWORD v17[2]; // [rsp+0h] [rbp-160h] BYREF
  void *src; // [rsp+10h] [rbp-150h] BYREF
  size_t n; // [rsp+18h] [rbp-148h]
  __int64 v20; // [rsp+20h] [rbp-140h]
  _BYTE v21[312]; // [rsp+28h] [rbp-138h] BYREF

  v5 = a1[37];
  v20 = 256;
  v17[0] = &src;
  v6 = *(_QWORD *)(v5 + 16);
  v17[1] = 0;
  src = v21;
  n = 0;
  (*(void (__fastcall **)(__int64, __int64, void **, _QWORD *, __int64))(*(_QWORD *)v6 + 24LL))(v6, a2, &src, v17, a3);
  v7 = 0;
  result = sub_E8BB10(a1, 0);
  v11 = src;
  *(_QWORD *)(result + 32) = a3;
  v12 = *(_QWORD *)(result + 48);
  v13 = result;
  v14 = n;
  *(_BYTE *)(result + 29) |= 1u;
  if ( v14 + v12 > *(_QWORD *)(result + 56) )
  {
    v7 = (void *)(result + 64);
    result = sub_C8D290(result + 40, (const void *)(result + 64), v14 + v12, 1u, v9, v10);
    v12 = *(_QWORD *)(v13 + 48);
  }
  if ( v14 )
  {
    v7 = v11;
    result = (__int64)memcpy((void *)(*(_QWORD *)(v13 + 40) + v12), v11, v14);
    v12 = *(_QWORD *)(v13 + 48);
  }
  v15 = v12 + v14;
  v16 = src;
  *(_QWORD *)(v13 + 48) = v15;
  if ( v16 != v21 )
    result = _libc_free(v16, v7);
  if ( (void **)v17[0] != &src )
    return _libc_free(v17[0], v7);
  return result;
}
