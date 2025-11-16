// Function: sub_2E30D70
// Address: 0x2e30d70
//
__int64 __fastcall sub_2E30D70(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // r14
  _DWORD *v4; // rdx
  _BYTE *v5; // rax
  __int64 v6; // rdi
  const char *v7; // rdi
  const char *v8; // [rsp-108h] [rbp-108h] BYREF
  __int64 v9; // [rsp-100h] [rbp-100h]
  __int64 v10; // [rsp-F8h] [rbp-F8h]
  unsigned __int64 v11; // [rsp-F0h] [rbp-F0h]
  _DWORD *v12; // [rsp-E8h] [rbp-E8h]
  __int64 v13; // [rsp-E0h] [rbp-E0h]
  const char **v14; // [rsp-D8h] [rbp-D8h]
  const char *v15; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v16; // [rsp-C0h] [rbp-C0h]
  __int64 v17; // [rsp-B8h] [rbp-B8h]
  _BYTE v18[176]; // [rsp-B0h] [rbp-B0h] BYREF

  result = *(_QWORD *)(a1 + 272);
  if ( !result )
  {
    v13 = 0x100000000LL;
    v3 = *(_QWORD *)(a1 + 32);
    v14 = &v15;
    v15 = v18;
    v8 = (const char *)&unk_49DD288;
    v16 = 0;
    v17 = 128;
    v9 = 2;
    v10 = 0;
    v11 = 0;
    v12 = 0;
    sub_CB5980((__int64)&v8, 0, 0, 0);
    v4 = v12;
    if ( v11 - (unsigned __int64)v12 <= 6 )
    {
      sub_CB6200((__int64)&v8, "$ehgcr_", 7u);
    }
    else
    {
      *v12 = 1734894884;
      *((_WORD *)v4 + 2) = 29283;
      *((_BYTE *)v4 + 6) = 95;
      v12 = (_DWORD *)((char *)v12 + 7);
    }
    sub_CB59D0((__int64)&v8, *(unsigned int *)(v3 + 336));
    v5 = v12;
    if ( (unsigned __int64)v12 >= v11 )
    {
      sub_CB5D20((__int64)&v8, 95);
    }
    else
    {
      v12 = (_DWORD *)((char *)v12 + 1);
      *v5 = 95;
    }
    sub_CB59F0((__int64)&v8, *(int *)(a1 + 24));
    v8 = (const char *)&unk_49DD388;
    sub_CB5840((__int64)&v8);
    v6 = *(_QWORD *)(v3 + 24);
    LOWORD(v12) = 261;
    v8 = v15;
    v9 = v16;
    result = sub_E6C460(v6, &v8);
    v7 = v15;
    *(_QWORD *)(a1 + 272) = result;
    if ( v7 != v18 )
    {
      _libc_free((unsigned __int64)v7);
      return *(_QWORD *)(a1 + 272);
    }
  }
  return result;
}
