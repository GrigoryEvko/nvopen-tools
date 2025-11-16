// Function: sub_2E2C870
// Address: 0x2e2c870
//
__int64 __fastcall sub_2E2C870(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r13d
  _QWORD *v7; // r15
  _QWORD *v8; // r14
  int v10; // ecx
  int v12; // ebx
  char v13; // al
  unsigned __int64 v14; // r15
  __int64 result; // rax
  __int64 v16; // rdx
  __int64 i; // rcx
  int v20; // ecx
  _QWORD *v22; // rax
  unsigned __int64 v23; // rax
  char v24; // dl
  unsigned int v27; // r15d
  int v28; // r14d
  __int64 v29; // rsi
  _QWORD *v30; // rax
  __int64 v32; // [rsp+8h] [rbp-48h]
  __int64 v33; // [rsp+8h] [rbp-48h]
  _QWORD *v34; // [rsp+10h] [rbp-40h]
  __int64 v35; // [rsp+10h] [rbp-40h]

  v6 = *(_DWORD *)(a2 + 24);
  v32 = a4;
  v7 = (_QWORD *)(*(_QWORD *)a5 + 32LL * *(int *)(a4 + 24));
  v8 = (_QWORD *)*v7;
  v34 = v7;
  if ( (_QWORD *)*v7 == v7 )
  {
    v14 = 0;
    v12 = 0;
    v13 = 1;
  }
  else
  {
    _RAX = v8[3];
    a5 = (unsigned int)(*((_DWORD *)v8 + 4) << 7);
    if ( _RAX )
    {
      v10 = 0;
    }
    else
    {
      _RAX = v8[4];
      if ( !_RAX )
LABEL_44:
        BUG();
      v10 = 64;
    }
    __asm { tzcnt   rax, rax }
    a4 = (unsigned int)(_RAX + v10);
    v12 = a5 + a4;
    v13 = 0;
    v14 = v8[(((unsigned int)(a5 + a4) >> 6) & 1) + 3] >> a4;
  }
  if ( !v13 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v22 = (_QWORD *)sub_2E29D60(a1, v12 | 0x80000000, a3, a4, a5, a6);
        sub_FDE240(v22, v6);
        LODWORD(a4) = v12 + 1;
        v23 = v14 >> 1;
        if ( !(v14 >> 1) )
          break;
        while ( 1 )
        {
          v24 = v23;
          v14 = v23;
          v12 = a4;
          v23 >>= 1;
          a3 = v24 & 1;
          a4 = (unsigned int)(a4 + 1);
          if ( a3 )
            break;
          if ( !v23 )
            goto LABEL_21;
        }
      }
LABEL_21:
      a3 = a4 & 0x7F;
      if ( v8[((unsigned __int8)(a4 & 0x7F) >> 6) + 3] & (-1LL << a4) )
        break;
      if ( (unsigned __int8)(a4 & 0x7F) >> 6 != 1 && (_RAX = v8[4]) != 0 )
      {
        __asm { tzcnt   rax, rax }
        a4 = (unsigned int)(_RAX + 64);
LABEL_23:
        if ( !(_DWORD)a3 )
          goto LABEL_13;
        a5 = (unsigned int)(*((_DWORD *)v8 + 4) << 7);
        v12 = a5 + a4;
        v14 = v8[((unsigned int)a4 >> 6) + 3] >> a4;
      }
      else
      {
LABEL_13:
        v8 = (_QWORD *)*v8;
        if ( v34 == v8 )
          goto LABEL_7;
        _RAX = v8[3];
        a5 = (unsigned int)(*((_DWORD *)v8 + 4) << 7);
        if ( _RAX )
        {
          v20 = 0;
        }
        else
        {
          _RAX = v8[4];
          if ( !_RAX )
            goto LABEL_44;
          v20 = 64;
        }
        __asm { tzcnt   rax, rax }
        a4 = (unsigned int)(_RAX + v20);
        v12 = a5 + a4;
        v14 = v8[(((unsigned int)(a5 + a4) >> 6) & 1) + 3] >> a4;
      }
    }
    __asm { tzcnt   rax, rax }
    a4 = (unsigned int)_RAX + (a4 & 0x40);
    goto LABEL_23;
  }
LABEL_7:
  result = v32;
  v16 = *(_QWORD *)(v32 + 56);
  for ( i = v32 + 48; i != v16; v16 = *(_QWORD *)(v16 + 8) )
  {
    result = *(unsigned __int16 *)(v16 + 68);
    if ( *(_WORD *)(v16 + 68) && (_DWORD)result != 68 )
      break;
    v27 = 1;
    v28 = *(_DWORD *)(v16 + 40) & 0xFFFFFF;
    if ( v28 != 1 )
    {
      do
      {
        while ( 1 )
        {
          v29 = *(_QWORD *)(v16 + 32);
          result = 5LL * (v27 + 1);
          if ( a2 == *(_QWORD *)(v29 + 40LL * (v27 + 1) + 24) )
          {
            result = v29 + 40LL * v27;
            if ( (*(_BYTE *)(result + 4) & 1) == 0
              && (*(_BYTE *)(result + 4) & 2) == 0
              && ((*(_BYTE *)(result + 3) & 0x10) == 0 || (*(_DWORD *)result & 0xFFF00) != 0) )
            {
              break;
            }
          }
          v27 += 2;
          if ( v27 == v28 )
            goto LABEL_35;
        }
        v33 = i;
        v27 += 2;
        v35 = v16;
        v30 = (_QWORD *)sub_2E29D60(a1, *(_DWORD *)(result + 8), v16, i, a5, a6);
        result = sub_FDE240(v30, v6);
        i = v33;
        v16 = v35;
      }
      while ( v27 != v28 );
    }
LABEL_35:
    if ( (*(_BYTE *)v16 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v16 + 44) & 8) != 0 )
        v16 = *(_QWORD *)(v16 + 8);
    }
  }
  return result;
}
