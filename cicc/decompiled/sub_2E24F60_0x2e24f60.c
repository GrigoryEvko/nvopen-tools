// Function: sub_2E24F60
// Address: 0x2e24f60
//
unsigned __int64 __fastcall sub_2E24F60(_QWORD *a1, __int64 a2)
{
  __m128i *v4; // rdx
  __m128i si128; // xmm0
  _QWORD *v6; // r13
  int v8; // ecx
  int v10; // ecx
  unsigned int v11; // r14d
  char v12; // al
  unsigned __int64 v13; // r15
  void *v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // r14
  unsigned __int64 result; // rax
  __int64 v18; // r14
  unsigned __int64 v19; // r13
  __int64 v20; // rdi
  __int64 v21; // rax
  _WORD *v22; // rdx
  int v23; // r10d
  __int64 v24; // rdi
  int v27; // ecx
  int v29; // ecx
  __int64 v30; // rax
  _WORD *v31; // rdx
  unsigned int v32; // ecx
  unsigned __int64 v33; // rax
  char v34; // dl
  unsigned int v35; // edx
  unsigned int v38; // ecx
  __m128i v40; // xmm0
  size_t v41; // rdx
  char *v42; // rsi
  unsigned __int64 v43; // [rsp-10h] [rbp-40h]

  v4 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v4 <= 0x12u )
  {
    sub_CB6200(a2, "  Alive in blocks: ", 0x13u);
    v6 = (_QWORD *)*a1;
    if ( a1 != (_QWORD *)*a1 )
      goto LABEL_3;
LABEL_45:
    v13 = 0;
    v11 = 0;
    v12 = 1;
    goto LABEL_7;
  }
  si128 = _mm_load_si128((const __m128i *)&xmmword_444FE00);
  v4[1].m128i_i8[2] = 32;
  v4[1].m128i_i16[0] = 14963;
  *v4 = si128;
  *(_QWORD *)(a2 + 32) += 19LL;
  v6 = (_QWORD *)*a1;
  if ( a1 == (_QWORD *)*a1 )
    goto LABEL_45;
LABEL_3:
  _RAX = v6[3];
  if ( _RAX )
  {
    v8 = 0;
  }
  else
  {
    _RAX = v6[4];
    if ( !_RAX )
LABEL_49:
      BUG();
    v8 = 64;
  }
  __asm { tzcnt   rax, rax }
  v10 = _RAX + v8;
  v11 = v10 + (*((_DWORD *)v6 + 4) << 7);
  v12 = 0;
  v13 = v6[((v11 >> 6) & 1) + 3] >> v10;
LABEL_7:
  if ( !v12 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v30 = sub_CB59D0(a2, v11);
        v31 = *(_WORD **)(v30 + 32);
        if ( *(_QWORD *)(v30 + 24) - (_QWORD)v31 <= 1u )
        {
          sub_CB6200(v30, (unsigned __int8 *)", ", 2u);
        }
        else
        {
          *v31 = 8236;
          *(_QWORD *)(v30 + 32) += 2LL;
        }
        v32 = v11 + 1;
        v33 = v13 >> 1;
        if ( !(v13 >> 1) )
          break;
        while ( 1 )
        {
          v34 = v33;
          v13 = v33;
          v11 = v32;
          v33 >>= 1;
          ++v32;
          if ( (v34 & 1) != 0 )
            break;
          if ( !v33 )
            goto LABEL_30;
        }
      }
LABEL_30:
      v35 = v32 & 0x7F;
      if ( v6[(v35 >> 6) + 3] & (-1LL << v32) )
        break;
      if ( (unsigned __int8)(v32 & 0x7F) >> 6 != 1 && (_RAX = v6[4]) != 0 )
      {
        __asm { tzcnt   rax, rax }
        v38 = _RAX + 64;
LABEL_32:
        if ( !v35 )
          goto LABEL_20;
        v11 = v38 + (*((_DWORD *)v6 + 4) << 7);
        v13 = v6[(v38 >> 6) + 3] >> v38;
      }
      else
      {
LABEL_20:
        v6 = (_QWORD *)*v6;
        if ( a1 == v6 )
          goto LABEL_8;
        _RAX = v6[3];
        if ( _RAX )
        {
          v27 = 0;
        }
        else
        {
          _RAX = v6[4];
          if ( !_RAX )
            goto LABEL_49;
          v27 = 64;
        }
        __asm { tzcnt   rax, rax }
        v29 = _RAX + v27;
        v11 = v29 + (*((_DWORD *)v6 + 4) << 7);
        v13 = v6[((v11 >> 6) & 1) + 3] >> v29;
      }
    }
    __asm { tzcnt   rax, rax }
    v38 = _RAX + (v32 & 0x40);
    goto LABEL_32;
  }
LABEL_8:
  v14 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v14 <= 0xCu )
  {
    sub_CB6200(a2, "\n  Killed by:", 0xDu);
    v16 = a1[5];
    result = a1[4];
    v15 = *(_QWORD *)(a2 + 32);
    if ( v16 != result )
      goto LABEL_10;
  }
  else
  {
    qmemcpy(v14, "\n  Killed by:", 13);
    v15 = *(_QWORD *)(a2 + 32) + 13LL;
    *(_QWORD *)(a2 + 32) = v15;
    v16 = a1[5];
    result = a1[4];
    if ( v16 != result )
    {
LABEL_10:
      v18 = (__int64)(v16 - result) >> 3;
      if ( (_DWORD)v18 )
      {
        v19 = 0;
        do
        {
          if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v15) > 5 )
          {
            *(_DWORD *)v15 = 538976266;
            v20 = a2;
            *(_WORD *)(v15 + 4) = 8992;
            *(_QWORD *)(a2 + 32) += 6LL;
          }
          else
          {
            v20 = sub_CB6200(a2, "\n    #", 6u);
          }
          v21 = sub_CB59D0(v20, v19);
          v22 = *(_WORD **)(v21 + 32);
          v23 = v21;
          if ( *(_QWORD *)(v21 + 24) - (_QWORD)v22 <= 1u )
          {
            v23 = sub_CB6200(v21, (unsigned __int8 *)": ", 2u);
          }
          else
          {
            *v22 = 8250;
            *(_QWORD *)(v21 + 32) += 2LL;
          }
          v24 = *(_QWORD *)(a1[4] + 8 * v19++);
          sub_2E91850(v24, v23, 1, 0, 0, 1, 0);
          result = v43;
          v15 = *(_QWORD *)(a2 + 32);
        }
        while ( (unsigned int)v18 != v19 );
      }
      if ( v15 != *(_QWORD *)(a2 + 24) )
      {
        *(_BYTE *)v15 = 10;
        ++*(_QWORD *)(a2 + 32);
        return result;
      }
      v41 = 1;
      v42 = "\n";
      return sub_CB6200(a2, (unsigned __int8 *)v42, v41);
    }
  }
  result = *(_QWORD *)(a2 + 24) - v15;
  if ( result > 0x12 )
  {
    v40 = _mm_load_si128((const __m128i *)&xmmword_444FE10);
    *(_BYTE *)(v15 + 18) = 10;
    *(_WORD *)(v15 + 16) = 2606;
    *(__m128i *)v15 = v40;
    *(_QWORD *)(a2 + 32) += 19LL;
    return result;
  }
  v41 = 19;
  v42 = " No instructions.\n\n";
  return sub_CB6200(a2, (unsigned __int8 *)v42, v41);
}
