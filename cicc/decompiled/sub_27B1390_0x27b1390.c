// Function: sub_27B1390
// Address: 0x27b1390
//
__int64 __fastcall sub_27B1390(__int64 a1, const void **a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  int v7; // ebx
  __int64 v8; // rdx
  int v9; // ebx
  int v10; // eax
  __int64 v11; // r10
  unsigned int v12; // r9d
  size_t v13; // r11
  char v14; // al
  int v16; // eax
  __int64 v17; // rdx
  size_t v18; // rdx
  int v19; // eax
  unsigned int v20; // r9d
  size_t v21; // [rsp+0h] [rbp-C0h]
  __int64 v22; // [rsp+0h] [rbp-C0h]
  size_t v23; // [rsp+0h] [rbp-C0h]
  __int64 v24; // [rsp+8h] [rbp-B8h]
  unsigned int v25; // [rsp+8h] [rbp-B8h]
  __int64 v26; // [rsp+8h] [rbp-B8h]
  unsigned int v27; // [rsp+10h] [rbp-B0h]
  size_t v28; // [rsp+10h] [rbp-B0h]
  unsigned int v29; // [rsp+10h] [rbp-B0h]
  int v30; // [rsp+1Ch] [rbp-A4h]
  __int64 v31; // [rsp+28h] [rbp-98h]
  unsigned __int64 v32[2]; // [rsp+30h] [rbp-90h] BYREF
  _BYTE v33[32]; // [rsp+40h] [rbp-80h] BYREF
  unsigned __int64 v34[2]; // [rsp+60h] [rbp-60h] BYREF
  _BYTE v35[80]; // [rsp+70h] [rbp-50h] BYREF

  v6 = 0;
  v7 = *(_DWORD *)(a1 + 24);
  if ( v7 )
  {
    v31 = *(_QWORD *)(a1 + 8);
    if ( !byte_4FFC580 && (unsigned int)sub_2207590((__int64)&byte_4FFC580) )
    {
      qword_4FFC5B0 = 0;
      qword_4FFC5A0 = (__int64)&qword_4FFC5B0;
      qword_4FFC5D0 = (__int64)algn_4FFC5E0;
      qword_4FFC5D8 = 0x400000000LL;
      qword_4FFC5A8 = 0x400000001LL;
      __cxa_atexit((void (*)(void *))sub_27ABC80, &qword_4FFC5A0, &qword_4A427C0);
      sub_2207640((__int64)&byte_4FFC580);
    }
    v8 = (unsigned int)qword_4FFC5A8;
    v32[0] = (unsigned __int64)v33;
    v32[1] = 0x400000000LL;
    if ( (_DWORD)qword_4FFC5A8 )
      sub_27ABF90((__int64)v32, (__int64)&qword_4FFC5A0, (unsigned int)qword_4FFC5A8, a4, a5, a6);
    v34[1] = 0x400000000LL;
    v34[0] = (unsigned __int64)v35;
    if ( (_DWORD)qword_4FFC5D8 )
      sub_27AC1D0((__int64)v34, (__int64)&qword_4FFC5D0, v8, a4, a5, a6);
    v9 = v7 - 1;
    v10 = sub_27B0000(*a2, (__int64)*a2 + 8 * *((unsigned int *)a2 + 2));
    v11 = *((unsigned int *)a2 + 2);
    v30 = 1;
    v12 = v9 & v10;
    v13 = 8 * v11;
    while ( 1 )
    {
      v6 = v31 + 96LL * v12;
      if ( v11 == *(_DWORD *)(v6 + 8) )
      {
        if ( !v13
          || (v22 = v11,
              v25 = v12,
              v28 = v13,
              v16 = memcmp(*a2, *(const void **)v6, v13),
              v13 = v28,
              v12 = v25,
              v11 = v22,
              !v16) )
        {
          v17 = *((unsigned int *)a2 + 14);
          if ( v17 == *(_DWORD *)(v6 + 56) )
          {
            v18 = 8 * v17;
            v23 = v13;
            v26 = v11;
            v29 = v12;
            if ( !v18 )
              break;
            v19 = memcmp(a2[6], *(const void **)(v6 + 48), v18);
            v12 = v29;
            v11 = v26;
            v13 = v23;
            if ( !v19 )
              break;
          }
        }
      }
      v21 = v13;
      v24 = v11;
      v27 = v12;
      v14 = sub_27ABCC0(v6, (__int64)v32);
      v11 = v24;
      v13 = v21;
      if ( v14 )
      {
        v6 = 0;
        break;
      }
      v20 = v30 + v27;
      ++v30;
      v12 = v9 & v20;
    }
    if ( (_BYTE *)v34[0] != v35 )
      _libc_free(v34[0]);
    if ( (_BYTE *)v32[0] != v33 )
      _libc_free(v32[0]);
  }
  return v6;
}
