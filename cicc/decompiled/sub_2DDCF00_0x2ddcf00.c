// Function: sub_2DDCF00
// Address: 0x2ddcf00
//
void __fastcall sub_2DDCF00(__int64 a1, __int64 a2)
{
  _BYTE *v3; // rsi
  __int64 v4; // rdx
  __int64 v5; // rdx
  size_t v6; // r15
  int v7; // r9d
  __int64 *v8; // [rsp+8h] [rbp-148h]
  _QWORD *v9; // [rsp+18h] [rbp-138h] BYREF
  __int64 *v10; // [rsp+20h] [rbp-130h] BYREF
  int v11[2]; // [rsp+28h] [rbp-128h]
  __int64 v12; // [rsp+30h] [rbp-120h] BYREF
  void *src[2]; // [rsp+40h] [rbp-110h] BYREF
  _QWORD v14[6]; // [rsp+60h] [rbp-F0h] BYREF
  __int64 *v15; // [rsp+90h] [rbp-C0h]
  _QWORD *v16; // [rsp+A0h] [rbp-B0h] BYREF
  _QWORD v17[6]; // [rsp+B0h] [rbp-A0h] BYREF
  unsigned __int64 v18[3]; // [rsp+E0h] [rbp-70h] BYREF
  _BYTE v19[88]; // [rsp+F8h] [rbp-58h] BYREF

  if ( sub_311A9A0(*(_QWORD *)(a1 + 8), 0) )
  {
    v18[0] = (unsigned __int64)v19;
    v14[5] = 0x100000000LL;
    v14[0] = &unk_49DD288;
    v15 = (__int64 *)v18;
    v18[1] = 0;
    v18[2] = 40;
    v14[1] = 2;
    memset(&v14[2], 0, 24);
    sub_CB5980((__int64)v14, 0, 0, 0);
    sub_3120780(v14, *(_QWORD *)(a1 + 8));
    sub_C7DA90(&v9, *v15, v15[1], "in-memory stable function map", (const char *)0x1D, 0);
    v3 = *(_BYTE **)(a2 + 232);
    v4 = *(_QWORD *)(a2 + 240);
    v16 = v17;
    sub_2DDB4F0((__int64 *)&v16, v3, (__int64)&v3[v4]);
    v5 = *(unsigned int *)(a2 + 284);
    v17[2] = *(_QWORD *)(a2 + 264);
    v17[3] = *(_QWORD *)(a2 + 272);
    v17[4] = *(_QWORD *)(a2 + 280);
    sub_3111C60(&v10, 1, v5, 1);
    v6 = *(_QWORD *)v11;
    v8 = v10;
    sub_C7EC60(src, v9);
    sub_2A41DE0((__int64 **)a2, v8, v6, 2u, (int)v8, v7, (char *)src[0], (size_t)src[1]);
    if ( v10 != &v12 )
      j_j___libc_free_0((unsigned __int64)v10);
    if ( v16 != v17 )
      j_j___libc_free_0((unsigned __int64)v16);
    if ( v9 )
      (*(void (__fastcall **)(_QWORD *))(*v9 + 8LL))(v9);
    v14[0] = &unk_49DD388;
    sub_CB5840((__int64)v14);
    if ( (_BYTE *)v18[0] != v19 )
      _libc_free(v18[0]);
  }
}
