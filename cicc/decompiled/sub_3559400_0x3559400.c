// Function: sub_3559400
// Address: 0x3559400
//
__int64 __fastcall sub_3559400(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char **v6; // r14
  __int64 v8; // rsi
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rax
  int v15; // eax
  char v16; // al
  _BYTE *v17; // rdi
  unsigned __int64 v19; // rdi
  _BYTE *v20; // rdx
  char v21; // al
  unsigned __int64 v22; // rdi
  __int64 v23; // [rsp+8h] [rbp-78h]
  __int64 v24; // [rsp+10h] [rbp-70h]
  int v25; // [rsp+18h] [rbp-68h]
  _BYTE *v26; // [rsp+20h] [rbp-60h] BYREF
  __int64 v27; // [rsp+28h] [rbp-58h]
  _BYTE v28[4]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v29; // [rsp+34h] [rbp-4Ch]
  __int64 v30; // [rsp+3Ch] [rbp-44h]
  __int64 v31; // [rsp+48h] [rbp-38h]
  int v32; // [rsp+50h] [rbp-30h]

  v6 = (char **)(a1 + 32);
  v8 = 0;
  v10 = *(_QWORD *)(a1 + 8);
  v11 = *(unsigned int *)(a1 + 40);
  v26 = v28;
  ++*(_QWORD *)a1;
  v23 = v10;
  LODWORD(v10) = *(_DWORD *)(a1 + 16);
  *(_QWORD *)(a1 + 8) = 0;
  LODWORD(v24) = v10;
  HIDWORD(v24) = *(_DWORD *)(a1 + 20);
  LODWORD(v10) = *(_DWORD *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  v12 = 0;
  v25 = v10;
  v27 = 0;
  if ( (_DWORD)v11 )
  {
    sub_353DE10((__int64)&v26, v6, v11, a4, a5, a6);
    v12 = *(_QWORD *)(a1 + 8);
    v8 = 8LL * *(unsigned int *)(a1 + 24);
  }
  v28[0] = *(_BYTE *)(a1 + 48);
  v29 = *(_QWORD *)(a1 + 52);
  v30 = *(_QWORD *)(a1 + 60);
  v31 = *(_QWORD *)(a1 + 72);
  v32 = *(_DWORD *)(a1 + 80);
  sub_C7D6A0(v12, v8, 8);
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  ++*(_QWORD *)a1;
  v13 = *(_QWORD *)(a2 + 8);
  ++*(_QWORD *)a2;
  v14 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 8) = v13;
  LODWORD(v13) = *(_DWORD *)(a2 + 16);
  *(_QWORD *)(a2 + 8) = v14;
  LODWORD(v14) = *(_DWORD *)(a1 + 16);
  *(_DWORD *)(a1 + 16) = v13;
  LODWORD(v13) = *(_DWORD *)(a2 + 20);
  *(_DWORD *)(a2 + 16) = v14;
  LODWORD(v14) = *(_DWORD *)(a1 + 20);
  *(_DWORD *)(a1 + 20) = v13;
  LODWORD(v13) = *(_DWORD *)(a2 + 24);
  *(_DWORD *)(a2 + 20) = v14;
  LODWORD(v14) = *(_DWORD *)(a1 + 24);
  *(_DWORD *)(a1 + 24) = v13;
  *(_DWORD *)(a2 + 24) = v14;
  if ( v6 != (char **)(a2 + 32) )
  {
    if ( *(_DWORD *)(a2 + 40) )
    {
      v22 = *(_QWORD *)(a1 + 32);
      if ( v22 != a1 + 48 )
        _libc_free(v22);
      *(_QWORD *)(a1 + 32) = *(_QWORD *)(a2 + 32);
      *(_DWORD *)(a1 + 40) = *(_DWORD *)(a2 + 40);
      *(_DWORD *)(a1 + 44) = *(_DWORD *)(a2 + 44);
      *(_QWORD *)(a2 + 32) = a2 + 48;
      *(_QWORD *)(a2 + 40) = 0;
    }
    else
    {
      *(_DWORD *)(a1 + 40) = 0;
    }
  }
  *(_BYTE *)(a1 + 48) = *(_BYTE *)(a2 + 48);
  *(_DWORD *)(a1 + 52) = *(_DWORD *)(a2 + 52);
  *(_DWORD *)(a1 + 56) = *(_DWORD *)(a2 + 56);
  *(_DWORD *)(a1 + 60) = *(_DWORD *)(a2 + 60);
  *(_DWORD *)(a1 + 64) = *(_DWORD *)(a2 + 64);
  *(_QWORD *)(a1 + 72) = *(_QWORD *)(a2 + 72);
  *(_DWORD *)(a1 + 80) = *(_DWORD *)(a2 + 80);
  sub_C7D6A0(*(_QWORD *)(a2 + 8), 8LL * *(unsigned int *)(a2 + 24), 8);
  ++*(_QWORD *)a2;
  *(_QWORD *)(a2 + 8) = v23;
  *(_QWORD *)(a2 + 16) = v24;
  *(_DWORD *)(a2 + 24) = v25;
  v15 = v27;
  if ( (_DWORD)v27 )
  {
    v19 = *(_QWORD *)(a2 + 32);
    if ( v19 != a2 + 48 )
    {
      _libc_free(v19);
      v15 = v27;
    }
    *(_DWORD *)(a2 + 40) = v15;
    v20 = v26;
    *(_DWORD *)(a2 + 44) = HIDWORD(v27);
    v21 = v28[0];
    *(_QWORD *)(a2 + 32) = v20;
    *(_BYTE *)(a2 + 48) = v21;
    *(_QWORD *)(a2 + 52) = v29;
    *(_QWORD *)(a2 + 60) = v30;
    *(_QWORD *)(a2 + 72) = v31;
    *(_DWORD *)(a2 + 80) = v32;
  }
  else
  {
    v16 = v28[0];
    v17 = v26;
    *(_DWORD *)(a2 + 40) = 0;
    *(_BYTE *)(a2 + 48) = v16;
    *(_QWORD *)(a2 + 52) = v29;
    *(_QWORD *)(a2 + 60) = v30;
    *(_QWORD *)(a2 + 72) = v31;
    *(_DWORD *)(a2 + 80) = v32;
    if ( v17 != v28 )
      _libc_free((unsigned __int64)v17);
  }
  return sub_C7D6A0(0, 0, 8);
}
