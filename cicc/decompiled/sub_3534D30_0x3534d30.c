// Function: sub_3534D30
// Address: 0x3534d30
//
void __fastcall sub_3534D30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  unsigned __int64 *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  unsigned __int64 v11; // r9
  _BYTE *v12; // rsi
  __int64 v13; // rdx
  int v14; // edx
  size_t v15; // r15
  int v16; // r9d
  unsigned __int64 *v17; // r12
  unsigned __int64 v18; // rdi
  __int64 *v19; // [rsp+8h] [rbp-148h]
  unsigned __int64 *v20; // [rsp+10h] [rbp-140h] BYREF
  _QWORD *v21; // [rsp+18h] [rbp-138h] BYREF
  __int64 *v22; // [rsp+20h] [rbp-130h] BYREF
  int v23[2]; // [rsp+28h] [rbp-128h]
  __int64 v24; // [rsp+30h] [rbp-120h] BYREF
  void *src[2]; // [rsp+40h] [rbp-110h] BYREF
  _QWORD v26[8]; // [rsp+60h] [rbp-F0h] BYREF
  _QWORD *v27; // [rsp+A0h] [rbp-B0h] BYREF
  _QWORD v28[6]; // [rsp+B0h] [rbp-A0h] BYREF
  _BYTE *v29; // [rsp+E0h] [rbp-70h] BYREF
  __int64 v30; // [rsp+E8h] [rbp-68h]
  __int64 v31; // [rsp+F0h] [rbp-60h]
  _BYTE v32[88]; // [rsp+F8h] [rbp-58h] BYREF

  if ( sub_3114900(*(unsigned __int64 **)(a1 + 200), 0, a3, a4, a5, a6) != 1 )
  {
    v29 = v32;
    v26[5] = 0x100000000LL;
    v26[0] = &unk_49DD288;
    v26[6] = &v29;
    v30 = 0;
    v31 = 40;
    v26[1] = 2;
    memset(&v26[2], 0, 24);
    sub_CB5980((__int64)v26, 0, 0, 0);
    v7 = *(unsigned __int64 **)(a1 + 200);
    *(_QWORD *)(a1 + 200) = 0;
    v20 = v7;
    sub_31166B0(&v20, (__int64)v26, v8, v9, v10, v11);
    sub_C7DA90(&v21, (__int64)v29, v30, "in-memory outlined hash tree", (const char *)0x1C, 0);
    v12 = *(_BYTE **)(a2 + 232);
    v13 = *(_QWORD *)(a2 + 240);
    v27 = v28;
    sub_3532860((__int64 *)&v27, v12, (__int64)&v12[v13]);
    v14 = *(_DWORD *)(a2 + 284);
    v28[2] = *(_QWORD *)(a2 + 264);
    v28[3] = *(_QWORD *)(a2 + 272);
    v28[4] = *(_QWORD *)(a2 + 280);
    sub_3111C60((__int64)&v22, 0, v14, 1);
    v15 = *(_QWORD *)v23;
    v19 = v22;
    sub_C7EC60(src, v21);
    sub_2A41DE0((__int64 **)a2, v19, v15, 0, (int)v19, v16, (char *)src[0], (size_t)src[1]);
    if ( v22 != &v24 )
      j_j___libc_free_0((unsigned __int64)v22);
    if ( v27 != v28 )
      j_j___libc_free_0((unsigned __int64)v27);
    if ( v21 )
      (*(void (__fastcall **)(_QWORD *))(*v21 + 8LL))(v21);
    v17 = v20;
    if ( v20 )
    {
      sub_3112140((__int64)(v20 + 2));
      v18 = v17[2];
      if ( (unsigned __int64 *)v18 != v17 + 8 )
        j_j___libc_free_0(v18);
      j_j___libc_free_0((unsigned __int64)v17);
    }
    v26[0] = &unk_49DD388;
    sub_CB5840((__int64)v26);
    if ( v29 != v32 )
      _libc_free((unsigned __int64)v29);
  }
}
