// Function: sub_31F0740
// Address: 0x31f0740
//
__int64 __fastcall sub_31F0740(_QWORD *a1, int a2, const void *a3, size_t a4, const void *a5, size_t a6)
{
  __int64 v6; // r14
  __int64 v7; // rax
  size_t v8; // r13
  __int64 v9; // rcx
  const void *v10; // r15
  __int64 v11; // rbx
  __int64 v12; // r12
  int v13; // eax
  int v14; // ecx
  _BYTE *v15; // rsi
  int v16; // eax
  __int64 result; // rax
  __int64 *v18; // rax
  __int64 v19; // [rsp+0h] [rbp-B0h]
  int v20[4]; // [rsp+Ch] [rbp-A4h] BYREF
  int v21; // [rsp+1Ch] [rbp-94h] BYREF
  _QWORD v22[4]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v23; // [rsp+40h] [rbp-70h]
  _QWORD v24[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v25; // [rsp+70h] [rbp-40h]

  v6 = (__int64)a1;
  v7 = a1[22];
  v20[0] = a2;
  v21 = 0;
  if ( *(_QWORD *)(v7 + 32) )
  {
    a4 = a6;
    a3 = a5;
  }
  v8 = a4;
  v9 = *((unsigned int *)a1 + 48);
  v10 = a3;
  if ( *((_DWORD *)a1 + 48) )
  {
    v11 = a1[23];
    v12 = 0;
    while ( 1 )
    {
      if ( v8 == *(_QWORD *)(v11 + 8) )
      {
        v19 = v9;
        if ( !v8 )
          break;
        a1 = *(_QWORD **)v11;
        v13 = memcmp(*(const void **)v11, v10, v8);
        v9 = v19;
        if ( !v13 )
          break;
      }
      ++v12;
      v11 += 48;
      if ( v9 == v12 )
        goto LABEL_17;
    }
    v14 = *(_DWORD *)(v11 + 40);
    v21 = v14;
  }
  else
  {
LABEL_17:
    v18 = sub_CEADF0();
    v25 = 770;
    a1 = (_QWORD *)v6;
    v23 = 1283;
    v22[0] = "Cannot find option named '";
    v24[0] = v22;
    v22[2] = v10;
    v22[3] = v8;
    v24[2] = "'!";
    result = sub_C53280(v6, (__int64)v24, 0, 0, (__int64)v18);
    if ( (_BYTE)result )
      return result;
    LOBYTE(v14) = v21;
  }
  v15 = *(_BYTE **)(v6 + 152);
  *(_DWORD *)(v6 + 136) |= 1 << v14;
  v16 = v20[0];
  *(_WORD *)(v6 + 14) = v20[0];
  if ( v15 == *(_BYTE **)(v6 + 160) )
  {
    a1 = (_QWORD *)(v6 + 144);
    sub_B8BBF0(v6 + 144, v15, v20);
  }
  else
  {
    if ( v15 )
    {
      *(_DWORD *)v15 = v16;
      v15 = *(_BYTE **)(v6 + 152);
    }
    v15 += 4;
    *(_QWORD *)(v6 + 152) = v15;
  }
  if ( !*(_QWORD *)(v6 + 600) )
    sub_4263D6(a1, v15, a3);
  (*(void (__fastcall **)(__int64, int *))(v6 + 608))(v6 + 584, &v21);
  return 0;
}
