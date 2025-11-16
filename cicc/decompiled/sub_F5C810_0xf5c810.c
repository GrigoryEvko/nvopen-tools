// Function: sub_F5C810
// Address: 0xf5c810
//
__int64 __fastcall sub_F5C810(char *a1, __int64 *a2, _QWORD *a3, __int64 a4)
{
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rsi
  int v10; // eax
  unsigned __int64 *v11; // rdi
  _QWORD *v12; // rdx
  unsigned __int64 v13; // rax
  void (__fastcall *v14)(_QWORD *, __int64, __int64); // rax
  __int64 *v15; // rsi
  _QWORD *v16; // r13
  _QWORD *v17; // r12
  __int64 v18; // rax
  _BYTE *v20; // [rsp+0h] [rbp-1F0h]
  _QWORD v22[2]; // [rsp+10h] [rbp-1E0h] BYREF
  char *v23; // [rsp+20h] [rbp-1D0h]
  __int64 v24; // [rsp+28h] [rbp-1C8h]
  _BYTE *v25; // [rsp+30h] [rbp-1C0h] BYREF
  __int64 v26; // [rsp+38h] [rbp-1B8h]
  _BYTE v27[432]; // [rsp+40h] [rbp-1B0h] BYREF

  v26 = 0x1000000000LL;
  v25 = v27;
  v22[0] = 6;
  v22[1] = 0;
  v23 = a1;
  if ( a1 != 0 && a1 + 4096 != 0 && a1 != (char *)-8192LL )
  {
    sub_BD73F0((__int64)v22);
    v8 = (unsigned int)v26;
    v9 = (unsigned int)v26 + 1LL;
    v10 = v26;
    if ( v9 > HIDWORD(v26) )
    {
      if ( v25 <= (_BYTE *)v22 )
      {
        v8 = (unsigned __int64)&v25[24 * (unsigned int)v26];
        if ( (unsigned __int64)v22 < v8 )
        {
          v20 = v25;
          sub_F39130((__int64)&v25, v9, v8, HIDWORD(v26), v6, v7);
          v10 = v26;
          v12 = &v25[(char *)v22 - v20];
          v11 = (unsigned __int64 *)&v25[24 * (unsigned int)v26];
LABEL_5:
          if ( !v11 )
            goto LABEL_12;
          goto LABEL_8;
        }
      }
      sub_F39130((__int64)&v25, v9, v8, HIDWORD(v26), v6, v7);
      v8 = (unsigned int)v26;
      v10 = v26;
    }
    v11 = (unsigned __int64 *)&v25[24 * v8];
    v12 = v22;
    goto LABEL_5;
  }
  v11 = (unsigned __int64 *)v27;
  v12 = v22;
LABEL_8:
  *v11 = 6;
  v13 = v12[2];
  v11[1] = 0;
  v11[2] = v13;
  if ( v13 != -4096 && v13 != 0 && v13 != -8192 )
    sub_BD6050(v11, *v12 & 0xFFFFFFFFFFFFFFF8LL);
  v10 = v26;
LABEL_12:
  LODWORD(v26) = v10 + 1;
  if ( v23 + 4096 != 0 && v23 != 0 && v23 != (char *)-8192LL )
    sub_BD60C0(v22);
  v23 = 0;
  v14 = *(void (__fastcall **)(_QWORD *, __int64, __int64))(a4 + 16);
  if ( v14 )
  {
    v14(v22, a4, 2);
    v24 = *(_QWORD *)(a4 + 24);
    v23 = *(char **)(a4 + 16);
  }
  v15 = a2;
  sub_F5C330((__int64)&v25, a2, a3, (__int64)v22);
  if ( v23 )
  {
    v15 = v22;
    ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v23)(v22, v22, 3);
  }
  v16 = v25;
  v17 = &v25[24 * (unsigned int)v26];
  if ( v25 != (_BYTE *)v17 )
  {
    do
    {
      v18 = *(v17 - 1);
      v17 -= 3;
      if ( v18 != -4096 && v18 != 0 && v18 != -8192 )
        sub_BD60C0(v17);
    }
    while ( v16 != v17 );
    v17 = v25;
  }
  if ( v17 != (_QWORD *)v27 )
    _libc_free(v17, v15);
  return 1;
}
