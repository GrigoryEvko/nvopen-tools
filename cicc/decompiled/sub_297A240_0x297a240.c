// Function: sub_297A240
// Address: 0x297a240
//
__int64 __fastcall sub_297A240(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  __int64 v7; // r14
  _QWORD *v8; // r13
  __int64 v9; // rax
  int v10; // r9d
  __int64 v11; // r8
  __int64 v12; // rbx
  int v13; // r12d
  __int64 v14; // rdi
  char v15; // al
  void *v16; // rsi
  void *v17; // r12
  __int64 v20; // [rsp+10h] [rbp-B0h]
  __int64 v21; // [rsp+18h] [rbp-A8h]
  char v22; // [rsp+26h] [rbp-9Ah]
  char v23; // [rsp+27h] [rbp-99h]
  __int64 v24; // [rsp+28h] [rbp-98h]
  __int64 v25; // [rsp+30h] [rbp-90h] BYREF
  _QWORD *v26; // [rsp+38h] [rbp-88h]
  int v27; // [rsp+40h] [rbp-80h]
  int v28; // [rsp+44h] [rbp-7Ch]
  int v29; // [rsp+48h] [rbp-78h]
  char v30; // [rsp+4Ch] [rbp-74h]
  _QWORD v31[2]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v32; // [rsp+60h] [rbp-60h] BYREF
  _BYTE *v33; // [rsp+68h] [rbp-58h]
  __int64 v34; // [rsp+70h] [rbp-50h]
  int v35; // [rsp+78h] [rbp-48h]
  char v36; // [rsp+7Ch] [rbp-44h]
  _BYTE v37[64]; // [rsp+80h] [rbp-40h] BYREF

  v6 = sub_BC1CD0(a4, &unk_4F81450, a3) + 8;
  v20 = a3;
  v7 = sub_BC1CD0(a4, &unk_4F875F0, a3) + 8;
  v8 = (_QWORD *)(sub_BC1CD0(a4, &unk_4F86540, a3) + 8);
  v9 = sub_BC1CD0(a4, &unk_4F8D468, v20);
  v22 = 0;
  v10 = *a2;
  v11 = v9 + 8;
  v21 = v20 + 72;
  while ( v21 != *(_QWORD *)(v20 + 80) )
  {
    v23 = 0;
    v12 = *(_QWORD *)(v20 + 80);
    v13 = v10;
    do
    {
      v14 = v12 - 24;
      if ( !v12 )
        v14 = 0;
      v24 = v11;
      v15 = sub_2978BA0(v14, v6, v7, v8, v11, v13);
      v12 = *(_QWORD *)(v12 + 8);
      v23 |= v15;
      v11 = v24;
    }
    while ( v21 != v12 );
    v10 = v13;
    if ( !v23 )
      break;
    v22 = v23;
  }
  v16 = (void *)(a1 + 32);
  v17 = (void *)(a1 + 80);
  if ( v22 )
  {
    v30 = 1;
    v26 = v31;
    v31[0] = &unk_4F82408;
    v27 = 2;
    v29 = 0;
    v32 = 0;
    v33 = v37;
    v34 = 2;
    v35 = 0;
    v36 = 1;
    v28 = 1;
    v25 = 1;
    sub_C8CF70(a1, v16, 2, (__int64)v31, (__int64)&v25);
    sub_C8CF70(a1 + 48, v17, 2, (__int64)v37, (__int64)&v32);
    if ( v36 )
    {
      if ( v30 )
        return a1;
    }
    else
    {
      _libc_free((unsigned __int64)v33);
      if ( v30 )
        return a1;
    }
    _libc_free((unsigned __int64)v26);
    return a1;
  }
  *(_QWORD *)(a1 + 8) = v16;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = v17;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  return a1;
}
