// Function: sub_25059C0
// Address: 0x25059c0
//
__int64 __fastcall sub_25059C0(__int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 *v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r15
  _QWORD *v10; // r13
  __int64 v11; // rax
  __int64 v12; // r14
  const char *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rcx
  void *v18; // rsi
  void *v19; // r12
  char i; // [rsp+42h] [rbp-9Eh]
  char v25; // [rsp+43h] [rbp-9Dh]
  char v26; // [rsp+44h] [rbp-9Ch]
  __int64 *v27; // [rsp+48h] [rbp-98h]
  __int64 v28; // [rsp+50h] [rbp-90h] BYREF
  _QWORD *v29; // [rsp+58h] [rbp-88h]
  int v30; // [rsp+60h] [rbp-80h]
  int v31; // [rsp+64h] [rbp-7Ch]
  int v32; // [rsp+68h] [rbp-78h]
  char v33; // [rsp+6Ch] [rbp-74h]
  _QWORD v34[2]; // [rsp+70h] [rbp-70h] BYREF
  __int64 v35; // [rsp+80h] [rbp-60h] BYREF
  _BYTE *v36; // [rsp+88h] [rbp-58h]
  __int64 v37; // [rsp+90h] [rbp-50h]
  int v38; // [rsp+98h] [rbp-48h]
  char v39; // [rsp+9Ch] [rbp-44h]
  _BYTE v40[64]; // [rsp+A0h] [rbp-40h] BYREF

  for ( i = 0; ; i = v25 )
  {
    v5 = sub_227ED20(a4, &qword_4FDADA8, (__int64 *)a3, a5);
    v6 = *(__int64 **)(a3 + 8);
    v7 = *(_QWORD *)(v5 + 8);
    v8 = *(unsigned int *)(a3 + 16);
    v27 = &v6[v8];
    if ( v6 == v27 )
      break;
    v25 = 0;
    v26 = (int)v8 > 1;
    do
    {
      while ( 1 )
      {
        v9 = *v6;
        v10 = *(_QWORD **)(*v6 + 8);
        v11 = sub_2503E40((__int64)v10, v7, *a2, v26);
        v12 = v11;
        if ( v11 )
          break;
        if ( v27 == ++v6 )
          goto LABEL_13;
      }
      sub_D2F240(*(__int64 **)a3, v9, v11);
      v13 = sub_BD5D20((__int64)v10);
      sub_BBB260(v7, (__int64)v10, (__int64)v13, v14);
      sub_B2E860(v10);
      v30 = 2;
      v32 = 0;
      v29 = v34;
      v33 = 1;
      v36 = v40;
      v35 = 0;
      v37 = 2;
      v38 = 0;
      v39 = 1;
      v31 = 1;
      v34[0] = &unk_4F82408;
      v28 = 1;
      v15 = *(_QWORD *)(v12 + 16);
      if ( v15 )
      {
        do
        {
          v16 = sub_B43CB0(*(_QWORD *)(v15 + 24));
          sub_BBE020(v7, v16, (__int64)&v28, v17);
          v15 = *(_QWORD *)(v15 + 8);
        }
        while ( v15 );
        if ( !v39 )
          _libc_free((unsigned __int64)v36);
        if ( !v33 )
          _libc_free((unsigned __int64)v29);
      }
      v25 = 1;
      ++v6;
    }
    while ( v27 != v6 );
LABEL_13:
    if ( !v25 )
      break;
  }
  v18 = (void *)(a1 + 32);
  v19 = (void *)(a1 + 80);
  if ( i )
  {
    v30 = 2;
    v29 = v34;
    v34[0] = &qword_4FDADA8;
    v32 = 0;
    v33 = 1;
    v35 = 0;
    v36 = v40;
    v37 = 2;
    v38 = 0;
    v39 = 1;
    v31 = 1;
    v28 = 1;
    if ( &qword_4FDADA8 != &qword_4F82400 && &qword_4FDADA8 != (__int64 *)&unk_4F82420 )
    {
      v31 = 2;
      v34[1] = &unk_4F82420;
      v28 = 2;
    }
    sub_C8CF70(a1, v18, 2, (__int64)v34, (__int64)&v28);
    sub_C8CF70(a1 + 48, v19, 2, (__int64)v40, (__int64)&v35);
    if ( v39 )
    {
      if ( v33 )
        return a1;
    }
    else
    {
      _libc_free((unsigned __int64)v36);
      if ( v33 )
        return a1;
    }
    _libc_free((unsigned __int64)v29);
    return a1;
  }
  *(_QWORD *)(a1 + 8) = v18;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = v19;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  return a1;
}
