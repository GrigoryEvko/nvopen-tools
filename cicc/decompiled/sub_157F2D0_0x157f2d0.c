// Function: sub_157F2D0
// Address: 0x157f2d0
//
char __fastcall sub_157F2D0(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // r8
  _QWORD *v4; // rax
  __int64 v5; // rbx
  char v7; // cl
  unsigned int v8; // r15d
  unsigned int v9; // ecx
  _QWORD *v10; // r14
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // r13
  unsigned int v16; // edi
  _QWORD *v17; // r14
  __int64 v18; // rax
  __int64 v19; // rdx
  _QWORD *v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rsi
  __int64 *v24; // rax
  __int64 v25; // rsi
  _QWORD *v26; // rbx
  unsigned __int64 *v27; // rcx
  unsigned __int64 v28; // rdx
  __int64 v29; // r8
  __int64 v31; // [rsp-40h] [rbp-40h]
  char v32; // [rsp-40h] [rbp-40h]

  v3 = a1 + 40;
  v4 = (_QWORD *)(*(_QWORD *)(a1 + 40) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (_QWORD *)(a1 + 40) == v4 )
    return (char)v4;
  v5 = *(_QWORD *)(a1 + 48);
  if ( !v5 )
    BUG();
  if ( *(_BYTE *)(v5 - 8) != 77 )
    return (char)v4;
  v7 = a3;
  v8 = *(_DWORD *)(v5 - 4) & 0xFFFFFFF;
  LOBYTE(v4) = v8 <= 2;
  if ( v8 == 2 )
  {
    v23 = v5 - 72;
    if ( (*(_BYTE *)(v5 - 1) & 0x40) != 0 )
      v23 = *(_QWORD *)(v5 - 32);
    v4 = (_QWORD *)(v23 + 24LL * *(unsigned int *)(v5 + 32) + 8);
    if ( a1 == v4[*v4 == a2] )
      goto LABEL_22;
    LOBYTE(v4) = 1;
  }
  if ( a3 != 1 && (_BYTE)v4 )
  {
    while ( 1 )
    {
      if ( !v5 )
        BUG();
      if ( *(_BYTE *)(v5 - 8) != 77 )
        return (char)v4;
      v9 = *(_DWORD *)(v5 - 4) & 0xFFFFFFF;
      if ( v9 )
      {
        v10 = (_QWORD *)(v5 - 24);
        v11 = 0;
        v12 = 24LL * *(unsigned int *)(v5 + 32) + 8;
        while ( 1 )
        {
          v13 = v5 - 24 - 24LL * v9;
          if ( (*(_BYTE *)(v5 - 1) & 0x40) != 0 )
            v13 = *(_QWORD *)(v5 - 32);
          if ( a2 == *(_QWORD *)(v13 + v12) )
            break;
          v11 = (unsigned int)(v11 + 1);
          v12 += 8;
          if ( v9 == (_DWORD)v11 )
          {
            v11 = 0xFFFFFFFFLL;
            break;
          }
        }
      }
      else
      {
        v11 = 0xFFFFFFFFLL;
        v10 = (_QWORD *)(v5 - 24);
      }
      v31 = v3;
      LOBYTE(v4) = sub_15F5350(v10, v11, 1);
      v3 = v31;
      if ( v8 == 2 )
      {
        if ( (*(_BYTE *)(v5 - 1) & 0x40) != 0 )
          v24 = *(__int64 **)(v5 - 32);
        else
          v24 = &v10[-3 * (*(_DWORD *)(v5 - 4) & 0xFFFFFFF)];
        v25 = *v24;
        if ( *v24 )
        {
          if ( v10 == (_QWORD *)v25 )
            v25 = sub_1599EF0(*v10);
        }
        sub_164D160(v10, v25);
        v26 = *(_QWORD **)(a1 + 48);
        sub_157EA20(v31, (__int64)(v26 - 3));
        v27 = (unsigned __int64 *)v26[1];
        v28 = *v26 & 0xFFFFFFFFFFFFFFF8LL;
        *v27 = v28 | *v27 & 7;
        *(_QWORD *)(v28 + 8) = v27;
        *v26 &= 7uLL;
        v26[1] = 0;
        LOBYTE(v4) = sub_164BEC0(v26 - 3, v26 - 3, v28, v27, v29);
        v5 = *(_QWORD *)(a1 + 48);
        v3 = v31;
      }
      else
      {
        v5 = *(_QWORD *)(a1 + 48);
      }
    }
  }
LABEL_22:
  while ( *(_BYTE *)(v5 - 8) == 77 )
  {
    v15 = *(_QWORD *)(v5 + 8);
    v16 = *(_DWORD *)(v5 - 4) & 0xFFFFFFF;
    if ( v16 )
    {
      v17 = (_QWORD *)(v5 - 24);
      v14 = 0;
      v18 = 24LL * *(unsigned int *)(v5 + 32) + 8;
      while ( 1 )
      {
        v19 = v5 - 24 - 24LL * v16;
        if ( (*(_BYTE *)(v5 - 1) & 0x40) != 0 )
          v19 = *(_QWORD *)(v5 - 32);
        if ( a2 == *(_QWORD *)(v19 + v18) )
          break;
        v14 = (unsigned int)(v14 + 1);
        v18 += 8;
        if ( v16 == (_DWORD)v14 )
        {
          v14 = 0xFFFFFFFFLL;
          break;
        }
      }
    }
    else
    {
      v14 = 0xFFFFFFFFLL;
      v17 = (_QWORD *)(v5 - 24);
    }
    v32 = v7;
    LOBYTE(v4) = sub_15F5350(v17, v14, 0);
    v7 = v32;
    if ( !v32 )
    {
      v4 = (_QWORD *)sub_15F5600(v17);
      v7 = 0;
      if ( v4 != v17 )
      {
        if ( v4 )
        {
          v20 = v4;
          sub_164D160(v17, v4);
          LOBYTE(v4) = sub_15F20C0(v17, v20, v21, v22);
          v7 = v32;
        }
      }
    }
    if ( !v15 )
      BUG();
    v5 = v15;
  }
  return (char)v4;
}
