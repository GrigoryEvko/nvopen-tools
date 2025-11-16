// Function: sub_253C360
// Address: 0x253c360
//
_BOOL8 __fastcall sub_253C360(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // eax
  int v8; // eax
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  _QWORD *v11; // rdi
  unsigned int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // r8
  char *v15; // rax
  _BOOL4 v16; // r12d
  char v18; // dl
  int v19; // [rsp+8h] [rbp-118h]
  int v20; // [rsp+Ch] [rbp-114h]
  __int64 v21; // [rsp+18h] [rbp-108h]
  __int64 v22; // [rsp+28h] [rbp-F8h] BYREF
  _QWORD v23[4]; // [rsp+30h] [rbp-F0h] BYREF
  _QWORD *v24; // [rsp+50h] [rbp-D0h] BYREF
  int v25; // [rsp+58h] [rbp-C8h]
  int v26; // [rsp+5Ch] [rbp-C4h]
  _QWORD v27[6]; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v28; // [rsp+90h] [rbp-90h] BYREF
  char *v29; // [rsp+98h] [rbp-88h]
  __int64 v30; // [rsp+A0h] [rbp-80h]
  int v31; // [rsp+A8h] [rbp-78h]
  char v32; // [rsp+ACh] [rbp-74h]
  char v33; // [rsp+B0h] [rbp-70h] BYREF

  v7 = *(_DWORD *)(a1 + 124);
  v32 = 1;
  v28 = 0;
  v20 = v7;
  v8 = *(_DWORD *)(a1 + 128);
  v30 = 8;
  v19 = v8;
  v29 = &v33;
  v24 = v27;
  v9 = *(_QWORD *)(a1 + 72);
  v31 = 0;
  v26 = 6;
  v10 = v9 & 0xFFFFFFFFFFFFFFFCLL;
  if ( (v9 & 3) == 3 )
    v10 = *(_QWORD *)(v10 + 24);
  v27[0] = v10;
  v11 = v27;
  v23[2] = &v24;
  v12 = 1;
  v25 = 1;
  v23[0] = a1;
  v23[1] = a2;
  v22 = a1;
  while ( 1 )
  {
    v13 = v12;
    v14 = v11[v12 - 1];
    v25 = v12 - 1;
    if ( v32 )
    {
      v15 = v29;
      a4 = HIDWORD(v30);
      v13 = (__int64)&v29[8 * HIDWORD(v30)];
      if ( v29 != (char *)v13 )
      {
        while ( v14 != *(_QWORD *)v15 )
        {
          v15 += 8;
          if ( (char *)v13 == v15 )
            goto LABEL_19;
        }
        goto LABEL_9;
      }
LABEL_19:
      if ( HIDWORD(v30) < (unsigned int)v30 )
        break;
    }
    v21 = v14;
    sub_C8CC70((__int64)&v28, v14, v13, a4, v14, a6);
    v14 = v21;
    if ( v18 )
      goto LABEL_17;
LABEL_9:
    v12 = v25;
    v11 = v24;
    if ( !v25 )
    {
      v16 = *(_DWORD *)(a1 + 124) - *(_DWORD *)(a1 + 128) == v20 - v19;
      goto LABEL_11;
    }
  }
  ++HIDWORD(v30);
  *(_QWORD *)v13 = v14;
  ++v28;
LABEL_17:
  if ( (unsigned __int8)sub_252FFB0(
                          a2,
                          (unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *))sub_254F480,
                          (__int64)v23,
                          a1,
                          v14,
                          1,
                          1,
                          1,
                          (unsigned __int8 (__fastcall *)(__int64, __int64, __int64))sub_254BB70,
                          (__int64)&v22) )
    goto LABEL_9;
  v11 = v24;
  v16 = 0;
  *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
LABEL_11:
  if ( v11 != v27 )
    _libc_free((unsigned __int64)v11);
  if ( !v32 )
    _libc_free((unsigned __int64)v29);
  return v16;
}
