// Function: sub_34AC2B0
// Address: 0x34ac2b0
//
void __fastcall sub_34AC2B0(__int64 *a1, const __m128i *a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r9
  __int64 v9; // r8
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r12
  __int64 v13; // rbx
  __int64 v14; // r13
  __int64 v15; // rax
  _QWORD *v16; // rax
  char *v17; // rdi
  __int64 v18; // [rsp+8h] [rbp-78h]
  char *v19; // [rsp+10h] [rbp-70h] BYREF
  int v20; // [rsp+18h] [rbp-68h]
  char v21; // [rsp+20h] [rbp-60h] BYREF
  unsigned __int64 v22[2]; // [rsp+30h] [rbp-50h] BYREF
  _BYTE v23[64]; // [rsp+40h] [rbp-40h] BYREF

  sub_34A0610(&v19, (_QWORD *)a1[3], (__int64)a2);
  sub_34A9810(a1[1], a2, v2, v3, v4, v5);
  v22[0] = (unsigned __int64)v23;
  v9 = a1[1];
  v22[1] = 0x200000000LL;
  if ( v20 )
  {
    v18 = v9;
    sub_349DD80((__int64)v22, (__int64)&v19, v6, v7, v9, v8);
    v9 = v18;
  }
  sub_34AADC0(v9, (__int64)v22, a2, v7, v9, v8);
  if ( (_BYTE *)v22[0] != v23 )
    _libc_free(v22[0]);
  v12 = a1[2];
  v13 = *a1;
  v14 = *(_QWORD *)&v19[8 * v20 - 8];
  v15 = *(unsigned int *)(v12 + 8);
  if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(v12 + 12) )
  {
    sub_C8D5F0(v12, (const void *)(v12 + 16), v15 + 1, 0x10u, v10, v11);
    v15 = *(unsigned int *)(v12 + 8);
  }
  v16 = (_QWORD *)(*(_QWORD *)v12 + 16 * v15);
  *v16 = v13;
  v17 = v19;
  v16[1] = v14;
  ++*(_DWORD *)(v12 + 8);
  if ( v17 != &v21 )
    _libc_free((unsigned __int64)v17);
}
