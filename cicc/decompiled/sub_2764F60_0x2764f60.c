// Function: sub_2764F60
// Address: 0x2764f60
//
__int64 __fastcall sub_2764F60(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r15
  __int64 v8; // r14
  __int64 v9; // rax
  bool v10; // r13
  void *v11; // r9
  void *v13; // rdx
  __int64 v14; // r8
  void **v15; // rsi
  __int64 *v16; // rdi
  __int64 **v17; // rax
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 i; // rcx
  __int64 v21; // rdx
  void **v22; // rax
  __int64 v23; // [rsp+8h] [rbp-A8h]
  __int64 v24; // [rsp+10h] [rbp-A0h]
  __int64 v25; // [rsp+18h] [rbp-98h]
  __int64 v26; // [rsp+20h] [rbp-90h] BYREF
  void **v27; // [rsp+28h] [rbp-88h]
  int v28; // [rsp+30h] [rbp-80h]
  int v29; // [rsp+34h] [rbp-7Ch]
  int v30; // [rsp+38h] [rbp-78h]
  char v31; // [rsp+3Ch] [rbp-74h]
  void *v32; // [rsp+40h] [rbp-70h] BYREF
  void *v33; // [rsp+48h] [rbp-68h] BYREF
  __int64 v34; // [rsp+50h] [rbp-60h] BYREF
  _BYTE *v35; // [rsp+58h] [rbp-58h]
  __int64 v36; // [rsp+60h] [rbp-50h]
  int v37; // [rsp+68h] [rbp-48h]
  char v38; // [rsp+6Ch] [rbp-44h]
  _BYTE v39[64]; // [rsp+70h] [rbp-40h] BYREF

  v24 = sub_BC1CD0(a4, &unk_4F86540, a3);
  v7 = sub_BC1CD0(a4, &unk_4F6D3F8, a3);
  v25 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v8 = *(_QWORD *)(sub_BC1CD0(a4, &unk_4F8F810, a3) + 8);
  v23 = sub_BC1CD0(a4, &unk_4F8FBC8, a3);
  v9 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  v10 = sub_2761E70(a3, v24 + 8, v8, v25 + 8, v23 + 8, v7 + 8, v9 + 8);
  if ( (unsigned __int8)sub_C92250() )
  {
    v18 = *(_QWORD *)(a3 + 80);
    v19 = a3 + 72;
    if ( a3 + 72 == v18 )
    {
      i = 0;
    }
    else
    {
      if ( !v18 )
        BUG();
      while ( 1 )
      {
        i = *(_QWORD *)(v18 + 32);
        if ( i != v18 + 24 )
          break;
        v18 = *(_QWORD *)(v18 + 8);
        if ( v19 == v18 )
          goto LABEL_2;
        if ( !v18 )
          BUG();
      }
    }
    while ( v19 != v18 )
    {
      for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v18 + 32) )
      {
        v21 = v18 - 24;
        if ( !v18 )
          v21 = 0;
        if ( i != v21 + 48 )
          break;
        v18 = *(_QWORD *)(v18 + 8);
        if ( v19 == v18 )
          goto LABEL_2;
        if ( !v18 )
          BUG();
      }
    }
  }
LABEL_2:
  v11 = (void *)(a1 + 32);
  if ( v10 )
  {
    v13 = &unk_4F82408;
    v31 = 1;
    v27 = &v32;
    v28 = 2;
    v30 = 0;
    v34 = 0;
    v35 = v39;
    v36 = 2;
    v37 = 0;
    v38 = 1;
    v29 = 1;
    v32 = &unk_4F82408;
    v26 = 1;
    if ( &unk_4F82408 == (_UNKNOWN *)&qword_4F82400 || &unk_4F82408 == &unk_4F8F810 )
    {
      v15 = &v33;
      v14 = 1;
    }
    else
    {
      v29 = 2;
      v14 = 2;
      v26 = 2;
      v15 = (void **)&v34;
      v33 = &unk_4F8F810;
    }
    v16 = (__int64 *)&unk_4F82408;
    v17 = (__int64 **)&v32;
    while ( v16 != &qword_4F82400 )
    {
      if ( ++v17 == (__int64 **)v15 )
      {
        v22 = &v32;
        while ( v13 != &unk_4F875F0 )
        {
          if ( ++v22 == v15 )
          {
            if ( (_DWORD)v14 == 1 )
            {
              v29 = 2;
              *v22 = &unk_4F875F0;
              ++v26;
            }
            else
            {
              sub_C8CC70((__int64)&v26, (__int64)&unk_4F875F0, (__int64)v13, (__int64)&v32, v14, (__int64)v11);
              v11 = (void *)(a1 + 32);
            }
            goto LABEL_12;
          }
          v13 = *v22;
        }
        break;
      }
      v16 = *v17;
    }
LABEL_12:
    sub_C8CF70(a1, v11, 2, (__int64)&v32, (__int64)&v26);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v39, (__int64)&v34);
    if ( !v38 )
      _libc_free((unsigned __int64)v35);
    if ( !v31 )
      _libc_free((unsigned __int64)v27);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v11;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  return a1;
}
