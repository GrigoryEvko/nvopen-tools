// Function: sub_BC7340
// Address: 0xbc7340
//
__int64 __fastcall sub_BC7340(_BYTE *a1, int a2, const void *a3, size_t a4, const void *a5, size_t a6)
{
  _BYTE *v6; // r14
  bool v7; // zf
  __int64 v8; // rax
  __int64 v9; // rax
  size_t v10; // r13
  __int64 v11; // rcx
  const void *v12; // r15
  __int64 v13; // rbx
  __int64 v14; // r12
  int v15; // eax
  _BYTE *v16; // rsi
  int v17; // eax
  _BYTE *v18; // rsi
  __int64 result; // rax
  __int64 v20; // rax
  __int64 v21; // [rsp+0h] [rbp-B0h]
  int v22[4]; // [rsp+Ch] [rbp-A4h] BYREF
  char v23; // [rsp+1Fh] [rbp-91h] BYREF
  _QWORD v24[4]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v25; // [rsp+40h] [rbp-70h]
  _QWORD v26[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v27; // [rsp+70h] [rbp-40h]

  v6 = a1;
  v7 = a1[184] == 0;
  v22[0] = a2;
  v23 = 0;
  if ( !v7 )
  {
    v8 = *((_QWORD *)a1 + 24);
    if ( v8 != *((_QWORD *)a1 + 25) )
      *((_QWORD *)a1 + 25) = v8;
    v9 = *((_QWORD *)a1 + 17);
    if ( v9 != *((_QWORD *)a1 + 18) )
      *((_QWORD *)a1 + 18) = v9;
    a1[184] = 0;
  }
  if ( *(_QWORD *)(*((_QWORD *)a1 + 28) + 32LL) )
  {
    a4 = a6;
    a3 = a5;
  }
  v10 = a4;
  v11 = *((unsigned int *)a1 + 60);
  v12 = a3;
  if ( *((_DWORD *)a1 + 60) )
  {
    v13 = *((_QWORD *)a1 + 29);
    v14 = 0;
    while ( 1 )
    {
      if ( v10 == *(_QWORD *)(v13 + 8) )
      {
        v21 = v11;
        if ( !v10 )
          break;
        a1 = *(_BYTE **)v13;
        v15 = memcmp(*(const void **)v13, v12, v10);
        v11 = v21;
        if ( !v15 )
          break;
      }
      ++v14;
      v13 += 48;
      if ( v11 == v14 )
        goto LABEL_27;
    }
    v23 = *(_BYTE *)(v13 + 40);
  }
  else
  {
LABEL_27:
    v20 = sub_CEADF0();
    v27 = 770;
    a1 = v6;
    v25 = 1283;
    v24[0] = "Cannot find option named '";
    v26[0] = v24;
    v24[2] = v12;
    v24[3] = v10;
    v26[2] = "'!";
    result = sub_C53280(v6, v26, 0, 0, v20);
    if ( (_BYTE)result )
      return result;
  }
  v16 = (_BYTE *)*((_QWORD *)v6 + 18);
  if ( v16 == *((_BYTE **)v6 + 19) )
  {
    a1 = v6 + 136;
    sub_931B30((__int64)(v6 + 136), v16, &v23);
  }
  else
  {
    if ( v16 )
    {
      *v16 = v23;
      v16 = (_BYTE *)*((_QWORD *)v6 + 18);
    }
    *((_QWORD *)v6 + 18) = v16 + 1;
  }
  v17 = v22[0];
  v18 = (_BYTE *)*((_QWORD *)v6 + 25);
  *((_WORD *)v6 + 7) = v22[0];
  if ( v18 == *((_BYTE **)v6 + 26) )
  {
    a1 = v6 + 192;
    sub_B8BBF0((__int64)(v6 + 192), v18, v22);
  }
  else
  {
    if ( v18 )
    {
      *(_DWORD *)v18 = v17;
      v18 = (_BYTE *)*((_QWORD *)v6 + 25);
    }
    v18 += 4;
    *((_QWORD *)v6 + 25) = v18;
  }
  if ( !*((_QWORD *)v6 + 81) )
    sub_4263D6(a1, v18, a3);
  (*((void (__fastcall **)(_BYTE *, char *, const void *, __int64))v6 + 82))(v6 + 632, &v23, a3, v11);
  return 0;
}
