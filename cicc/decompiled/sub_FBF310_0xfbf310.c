// Function: sub_FBF310
// Address: 0xfbf310
//
_QWORD *__fastcall sub_FBF310(__int64 a1, unsigned int a2, unsigned __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v6; // ebx
  __int64 v7; // r12
  __int64 v8; // rcx
  unsigned __int64 v9; // rax
  unsigned int v10; // r13d
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v15; // r13
  _QWORD *v16; // r15
  __int64 v17; // rax
  __int64 v18; // r9
  __int64 v19; // rdi
  _QWORD *v20; // rax
  __int64 v21; // rsi
  _QWORD *v22; // r12
  _QWORD *v23; // rdi
  __int64 v24; // rax
  __int64 v25; // [rsp+8h] [rbp-428h]
  __int64 v26; // [rsp+10h] [rbp-420h]
  __int64 v27; // [rsp+18h] [rbp-418h]
  __int64 v28; // [rsp+18h] [rbp-418h]
  _BYTE v29[1040]; // [rsp+20h] [rbp-410h] BYREF

  v6 = a2;
  v7 = *(_QWORD *)(a1 + 16);
  v8 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    if ( !(_BYTE)v8 )
    {
      v10 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    v15 = a1 + 16;
    v25 = a1 + 1008;
  }
  else
  {
    a3 = ((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
        | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1)) >> 16;
    v9 = (a3
        | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
        | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1))
       + 1;
    v6 = v9;
    if ( (unsigned int)v9 > 0x40 )
    {
      v15 = a1 + 16;
      v25 = a1 + 1008;
      if ( !(_BYTE)v8 )
      {
        v10 = *(_DWORD *)(a1 + 24);
        v11 = 248LL * (unsigned int)v9;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !(_BYTE)v8 )
      {
        v10 = *(_DWORD *)(a1 + 24);
        v6 = 64;
        v11 = 15872;
LABEL_5:
        v12 = sub_C7D670(v11, 8);
        *(_DWORD *)(a1 + 24) = v6;
        *(_QWORD *)(a1 + 16) = v12;
LABEL_8:
        v13 = 248LL * v10;
        sub_FBF050(a1, v7, v7 + v13);
        return (_QWORD *)sub_C7D6A0(v7, v13, 8);
      }
      v15 = a1 + 16;
      v6 = 64;
      v25 = a1 + 1008;
    }
  }
  v16 = v29;
  do
  {
    v17 = *(_QWORD *)v15;
    v18 = v15 + 248;
    if ( *(_QWORD *)v15 != -4096 && v17 != -8192 )
    {
      if ( v16 )
        *v16 = v17;
      v16[1] = 0;
      v19 = (__int64)(v16 + 1);
      v20 = v16 + 3;
      v16 += 31;
      *(v16 - 29) = 1;
      v21 = v15 + 8;
      do
      {
        if ( v20 )
          *v20 = -4096;
        v20 += 7;
      }
      while ( v20 != v16 );
      sub_FBED40(v19, (char **)v21, a3, v8, a5, v18);
      if ( (*(_BYTE *)(v15 + 16) & 1) != 0 )
      {
        v18 = v15 + 248;
        v22 = (_QWORD *)(v15 + 24);
        v8 = v15 + 248;
      }
      else
      {
        v22 = *(_QWORD **)(v15 + 24);
        v18 = v15 + 248;
        v21 = 56LL * *(unsigned int *)(v15 + 32);
        if ( !*(_DWORD *)(v15 + 32) )
          goto LABEL_34;
        v8 = (__int64)&v22[7 * *(unsigned int *)(v15 + 32)];
        v18 = v15 + 248;
        if ( v22 == (_QWORD *)v8 )
          goto LABEL_34;
      }
      do
      {
        if ( *v22 != -4096 && *v22 != -8192 )
        {
          v23 = (_QWORD *)v22[1];
          if ( v23 != v22 + 3 )
          {
            v26 = v18;
            v27 = v8;
            _libc_free(v23, v21);
            v18 = v26;
            v8 = v27;
          }
        }
        v22 += 7;
      }
      while ( v22 != (_QWORD *)v8 );
      if ( (*(_BYTE *)(v15 + 16) & 1) == 0 )
      {
        v22 = *(_QWORD **)(v15 + 24);
        v21 = 56LL * *(unsigned int *)(v15 + 32);
LABEL_34:
        v28 = v18;
        sub_C7D6A0((__int64)v22, v21, 8);
        v18 = v28;
      }
    }
    v15 = v18;
  }
  while ( v18 != v25 );
  if ( v6 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v24 = sub_C7D670(248LL * v6, 8);
    *(_DWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 16) = v24;
  }
  return sub_FBF050(a1, (__int64)v29, (__int64)v16);
}
