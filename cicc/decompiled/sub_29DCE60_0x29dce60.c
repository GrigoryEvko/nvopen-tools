// Function: sub_29DCE60
// Address: 0x29dce60
//
void __fastcall sub_29DCE60(__int64 a1, __int64 a2, char a3, __int64 a4)
{
  size_t v4; // r13
  const void *v5; // r14
  __int64 v6; // r12
  int v7; // eax
  int v8; // eax
  __int64 v9; // rax
  _QWORD v10[3]; // [rsp+0h] [rbp-A0h] BYREF
  bool v11; // [rsp+18h] [rbp-88h]
  char v12; // [rsp+19h] [rbp-87h]
  __int64 v13; // [rsp+20h] [rbp-80h]
  char *v14; // [rsp+28h] [rbp-78h]
  __int64 v15; // [rsp+30h] [rbp-70h]
  int v16; // [rsp+38h] [rbp-68h]
  char v17; // [rsp+3Ch] [rbp-64h]
  char v18; // [rsp+40h] [rbp-60h] BYREF
  __int64 v19; // [rsp+60h] [rbp-40h]
  __int64 v20; // [rsp+68h] [rbp-38h]
  __int64 v21; // [rsp+70h] [rbp-30h]
  unsigned int v22; // [rsp+78h] [rbp-28h]

  v10[0] = a1;
  v10[1] = a2;
  v10[2] = a4;
  v11 = 0;
  v12 = a3;
  v13 = 0;
  v14 = &v18;
  v15 = 4;
  v16 = 0;
  v17 = 1;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  if ( !a4 )
  {
    v4 = *(_QWORD *)(a1 + 176);
    v5 = *(const void **)(a1 + 168);
    v6 = *(_QWORD *)(a2 + 48) + 8LL * *(unsigned int *)(a2 + 56);
    v7 = sub_C92610();
    v8 = sub_C92860((__int64 *)(a2 + 48), v5, v4, v7);
    if ( v8 == -1 )
      v9 = *(_QWORD *)(a2 + 48) + 8LL * *(unsigned int *)(a2 + 56);
    else
      v9 = *(_QWORD *)(a2 + 48) + 8LL * v8;
    v11 = v9 != v6;
  }
  sub_29DCE50((unsigned int *)v10);
  sub_C7D6A0(v20, 16LL * v22, 8);
  if ( !v17 )
    _libc_free((unsigned __int64)v14);
}
