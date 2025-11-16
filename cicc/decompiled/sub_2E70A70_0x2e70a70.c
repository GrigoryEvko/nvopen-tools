// Function: sub_2E70A70
// Address: 0x2e70a70
//
__int64 __fastcall sub_2E70A70(__int64 a1, __int64 a2)
{
  int v3; // eax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r9
  __int64 v7; // rdi
  __int64 v8; // r8
  __int64 v9; // r15
  _BYTE *v10; // r12
  unsigned __int64 v11; // r13
  unsigned __int64 v12; // rdi
  __int64 v14; // rdx
  __int64 v15; // r15
  unsigned __int64 v16; // r8
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // [rsp+8h] [rbp-B8h]
  char *v19; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v20; // [rsp+18h] [rbp-A8h]
  char v21; // [rsp+20h] [rbp-A0h] BYREF
  _BYTE *v22; // [rsp+28h] [rbp-98h] BYREF
  __int64 v23; // [rsp+30h] [rbp-90h]
  _BYTE v24[56]; // [rsp+38h] [rbp-88h] BYREF
  __int64 v25; // [rsp+70h] [rbp-50h]
  __int64 v26; // [rsp+78h] [rbp-48h]
  char v27; // [rsp+80h] [rbp-40h]
  __int64 v28; // [rsp+84h] [rbp-3Ch]

  v23 = 0x600000000LL;
  v3 = *(_DWORD *)(a2 + 120);
  v19 = &v21;
  v20 = 0x100000000LL;
  v22 = v24;
  v25 = 0;
  v27 = 0;
  LODWORD(v28) = 0;
  v26 = a2;
  HIDWORD(v28) = v3;
  sub_2E708A0((__int64)&v19);
  v7 = a1 + 200;
  v8 = a1 + 224;
  if ( *(_BYTE *)(a1 + 328) )
  {
    sub_2E6C4A0(v7, &v19, v4, v5, v8, v6);
    sub_2E6E020(a1 + 224, (__int64)&v22);
    *(_QWORD *)(a1 + 296) = v25;
    *(_QWORD *)(a1 + 304) = v26;
    *(_BYTE *)(a1 + 312) = v27;
    *(_QWORD *)(a1 + 316) = v28;
    sub_2E6DCE0((__int64 *)&v22);
    v9 = (__int64)v22;
    v25 = 0;
    v26 = 0;
    v10 = &v22[8 * (unsigned int)v23];
    if ( v22 != v10 )
    {
      do
      {
        v11 = *((_QWORD *)v10 - 1);
        v10 -= 8;
        if ( v11 )
        {
          v12 = *(_QWORD *)(v11 + 24);
          if ( v12 != v11 + 40 )
            _libc_free(v12);
          j_j___libc_free_0(v11);
        }
      }
      while ( (_BYTE *)v9 != v10 );
      v10 = v22;
    }
    goto LABEL_9;
  }
  v14 = (unsigned int)v20;
  *(_QWORD *)(a1 + 208) = 0x100000000LL;
  *(_QWORD *)(a1 + 200) = a1 + 216;
  if ( (_DWORD)v14 )
  {
    sub_2E6C4A0(v7, &v19, v14, v5, v8, v6);
    v8 = a1 + 224;
  }
  *(_QWORD *)(a1 + 224) = a1 + 240;
  *(_QWORD *)(a1 + 232) = 0x600000000LL;
  if ( !(_DWORD)v23 )
  {
    v10 = v22;
    *(_QWORD *)(a1 + 296) = v25;
    *(_QWORD *)(a1 + 304) = v26;
    *(_BYTE *)(a1 + 312) = v27;
    *(_QWORD *)(a1 + 316) = v28;
LABEL_18:
    *(_BYTE *)(a1 + 328) = 1;
    goto LABEL_9;
  }
  sub_2E6E020(v8, (__int64)&v22);
  v15 = (__int64)v22;
  *(_QWORD *)(a1 + 296) = v25;
  *(_QWORD *)(a1 + 304) = v26;
  *(_BYTE *)(a1 + 312) = v27;
  *(_QWORD *)(a1 + 316) = v28;
  v10 = (_BYTE *)(v15 + 8LL * (unsigned int)v23);
  if ( (_BYTE *)v15 == v10 )
    goto LABEL_18;
  do
  {
    v16 = *((_QWORD *)v10 - 1);
    v10 -= 8;
    if ( v16 )
    {
      v17 = *(_QWORD *)(v16 + 24);
      if ( v17 != v16 + 40 )
      {
        v18 = v16;
        _libc_free(v17);
        v16 = v18;
      }
      j_j___libc_free_0(v16);
    }
  }
  while ( (_BYTE *)v15 != v10 );
  *(_BYTE *)(a1 + 328) = 1;
  v10 = v22;
LABEL_9:
  if ( v10 != v24 )
    _libc_free((unsigned __int64)v10);
  if ( v19 != &v21 )
    _libc_free((unsigned __int64)v19);
  return 0;
}
