// Function: sub_19F1B70
// Address: 0x19f1b70
//
void __fastcall sub_19F1B70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  char v5; // al
  __int64 *v6; // rdx
  unsigned int v7; // esi
  int v8; // eax
  int v9; // eax
  __int8 v10; // r14
  __m128i v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 *v17; // rax
  __int64 v18; // r8
  __int64 *v19; // r9
  int v21; // edx
  char v23; // cl
  unsigned int v24; // esi
  char v25; // al
  unsigned __int64 v26; // rdi
  unsigned int v27; // ecx
  unsigned __int64 v28; // rax
  char v29; // dl
  unsigned int v30; // edx
  unsigned int v33; // ecx
  __int64 v34; // rax
  int v37; // edx
  unsigned int v39; // ecx
  __int64 v41; // [rsp+8h] [rbp-58h] BYREF
  __int64 *v42; // [rsp+18h] [rbp-48h] BYREF
  __m128i v43; // [rsp+20h] [rbp-40h] BYREF

  v3 = a1 + 2200;
  v41 = a3;
  v43.m128i_i64[0] = a2;
  v43.m128i_i64[1] = a3;
  v5 = sub_19E8F30(a1 + 2200, v43.m128i_i64, &v42);
  v6 = v42;
  if ( v5 )
    return;
  v7 = *(_DWORD *)(a1 + 2224);
  v8 = *(_DWORD *)(a1 + 2216);
  ++*(_QWORD *)(a1 + 2200);
  v9 = v8 + 1;
  if ( 4 * v9 >= 3 * v7 )
  {
    v7 *= 2;
    goto LABEL_41;
  }
  if ( v7 - *(_DWORD *)(a1 + 2220) - v9 <= v7 >> 3 )
  {
LABEL_41:
    sub_19F1650(v3, v7);
    sub_19E8F30(v3, v43.m128i_i64, &v42);
    v6 = v42;
    v9 = *(_DWORD *)(a1 + 2216) + 1;
  }
  *(_DWORD *)(a1 + 2216) = v9;
  if ( *v6 != -8 || v6[1] != -8 )
    --*(_DWORD *)(a1 + 2220);
  *(__m128i *)v6 = _mm_loadu_si128(&v43);
  v11.m128i_i64[0] = (__int64)sub_1412190(a1 + 2232, v41);
  v10 = v11.m128i_i8[8];
  v11.m128i_i64[1] = *(_QWORD *)(a1 + 2248);
  if ( v11.m128i_i64[1] == *(_QWORD *)(a1 + 2240) )
    v12 = *(unsigned int *)(a1 + 2260);
  else
    v12 = *(unsigned int *)(a1 + 2256);
  v11.m128i_i64[1] += 8 * v12;
  v43 = v11;
  sub_19E4730((__int64)&v43);
  if ( v10 )
  {
    v34 = sub_19E56C0(a1 + 2360, v41);
    sub_19E1950((_QWORD *)(a1 + 2336), v34, HIDWORD(v34));
    return;
  }
  v13 = sub_14228C0(*(_QWORD *)(a1 + 32), v41);
  if ( v13 )
  {
    v14 = a1 + 2392;
    if ( (unsigned int)*(unsigned __int8 *)(v13 + 16) - 21 <= 1 )
      v13 = *(_QWORD *)(v13 + 72);
    v42 = (__int64 *)v13;
    if ( (unsigned __int8)sub_154CC80(v14, (__int64 *)&v42, &v43) )
    {
      v39 = *(_DWORD *)(v43.m128i_i64[0] + 8);
      v15 = 1LL << v39;
      v16 = 8LL * (v39 >> 6);
    }
    else
    {
      v15 = 1;
      v16 = 0;
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 2336) + v16) |= v15;
  }
  v17 = sub_19F1A30(a1 + 1864, &v41);
  v18 = v17[2];
  v19 = v17 + 2;
  if ( (__int64 *)v18 == v17 + 2 )
  {
    v26 = 0;
    v24 = 0;
    v25 = 1;
  }
  else
  {
    _RAX = *(_QWORD *)(v18 + 24);
    v21 = 0;
    if ( !_RAX )
    {
      _RAX = *(_QWORD *)(v18 + 32);
      v21 = 64;
      if ( !_RAX )
      {
        _RAX = *(_QWORD *)(v18 + 40);
        v21 = 128;
      }
    }
    __asm { tzcnt   rax, rax }
    v23 = v21 + _RAX;
    v24 = v21 + _RAX + (*(_DWORD *)(v18 + 16) << 7);
    v25 = 0;
    v26 = *(_QWORD *)(v18 + 8LL * ((v24 >> 6) & 1) + 24) >> v23;
  }
  if ( !v25 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v27 = v24 + 1;
        *(_QWORD *)(*(_QWORD *)(a1 + 2336) + 8LL * (v24 >> 6)) |= 1LL << v24;
        v28 = v26 >> 1;
        if ( !(v26 >> 1) )
          break;
        while ( 1 )
        {
          v29 = v28;
          v26 = v28;
          v24 = v27;
          v28 >>= 1;
          ++v27;
          if ( (v29 & 1) != 0 )
            break;
          if ( !v28 )
            goto LABEL_25;
        }
      }
LABEL_25:
      v30 = v27 & 0x7F;
      if ( *(_QWORD *)(v18 + 8LL * (v30 >> 6) + 24) & (-1LL << v27) )
        break;
      if ( (unsigned __int8)(v27 & 0x7F) >> 6 != 1 && (_RAX = *(_QWORD *)(v18 + 32)) != 0 )
      {
        __asm { tzcnt   rax, rax }
        v33 = _RAX + 64;
LABEL_27:
        if ( !v30 )
          goto LABEL_32;
        v24 = v33 + (*(_DWORD *)(v18 + 16) << 7);
        v26 = *(_QWORD *)(v18 + 8LL * (v33 >> 6) + 24) >> v33;
      }
      else
      {
LABEL_32:
        v18 = *(_QWORD *)v18;
        if ( v19 == (__int64 *)v18 )
          return;
        _RAX = *(_QWORD *)(v18 + 24);
        v37 = 0;
        if ( !_RAX )
        {
          _RAX = *(_QWORD *)(v18 + 32);
          v37 = 64;
          if ( !_RAX )
          {
            _RAX = *(_QWORD *)(v18 + 40);
            v37 = 128;
          }
        }
        __asm { tzcnt   rax, rax }
        v24 = v37 + _RAX + (*(_DWORD *)(v18 + 16) << 7);
        v26 = *(_QWORD *)(v18 + 8LL * ((v24 >> 6) & 1) + 24) >> ((unsigned __int8)v37 + (unsigned __int8)_RAX);
      }
    }
    __asm { tzcnt   rax, rax }
    v33 = _RAX + (v27 & 0x40);
    goto LABEL_27;
  }
}
