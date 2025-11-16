// Function: sub_A78730
// Address: 0xa78730
//
__int64 __fastcall sub_A78730(_QWORD *a1, const void *a2, size_t a3, const void *a4, size_t a5)
{
  _QWORD *v8; // r15
  _QWORD *v9; // rsi
  __int64 v10; // rax
  _QWORD *v11; // r9
  __int64 v12; // r8
  __int64 v14; // rax
  __int64 v15; // rsi
  unsigned __int64 v16; // r8
  unsigned __int64 v17; // r15
  int v18; // eax
  __int64 v19; // rax
  _QWORD *v20; // [rsp+8h] [rbp-E8h]
  _QWORD *v21; // [rsp+8h] [rbp-E8h]
  __int64 v23; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v24; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v25; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v26; // [rsp+18h] [rbp-D8h]
  __int64 v27; // [rsp+28h] [rbp-C8h] BYREF
  _QWORD v28[2]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE v29[176]; // [rsp+40h] [rbp-B0h] BYREF

  v8 = (_QWORD *)*a1;
  v28[0] = v29;
  v28[1] = 0x2000000000LL;
  sub_C653C0(v28, a2);
  if ( a5 )
    sub_C653C0(v28, a4);
  v9 = v28;
  v10 = sub_C65B40(v8 + 50, v28, &v27, off_49D9AB0);
  v11 = v8 + 50;
  v12 = v10;
  if ( !v10 )
  {
    v14 = v8[330];
    v15 = a3 + a5 + 26;
    v8[340] += v15;
    v16 = (v14 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v8[331] >= v15 + v16 && v14 )
    {
      v8[330] = v15 + v16;
    }
    else
    {
      v19 = sub_9D1E70((__int64)(v8 + 330), v15, a3 + a5 + 26, 3);
      v11 = v8 + 50;
      v16 = v19;
    }
    *(_QWORD *)v16 = 0;
    v17 = v16 + 24;
    *(_BYTE *)(v16 + 8) = 2;
    *(_DWORD *)(v16 + 12) = a3;
    *(_DWORD *)(v16 + 16) = a5;
    if ( a3 )
    {
      v20 = v11;
      v25 = v16;
      memmove((void *)(v16 + 24), a2, a3);
      v11 = v20;
      v16 = v25;
    }
    *(_BYTE *)(v17 + (unsigned int)a3) = 0;
    v18 = *(_DWORD *)(v16 + 12);
    if ( a5 )
    {
      v21 = v11;
      v26 = v16;
      memmove((void *)(v17 + (unsigned int)(v18 + 1)), a4, a5);
      v16 = v26;
      v11 = v21;
      v18 = *(_DWORD *)(v26 + 12);
    }
    v9 = (_QWORD *)v16;
    v24 = v16;
    *(_BYTE *)(v17 + (unsigned int)(v18 + *(_DWORD *)(v16 + 16) + 1)) = 0;
    sub_C657C0(v11, v16, v27, off_49D9AB0);
    v12 = v24;
  }
  if ( (_BYTE *)v28[0] != v29 )
  {
    v23 = v12;
    _libc_free(v28[0], v9);
    return v23;
  }
  return v12;
}
