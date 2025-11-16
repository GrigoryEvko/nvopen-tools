// Function: sub_B4F6B0
// Address: 0xb4f6b0
//
__int64 __fastcall sub_B4F6B0(__int64 a1, unsigned __int64 a2, signed int a3)
{
  unsigned int v3; // r8d
  unsigned int v5; // r10d
  unsigned int v6; // r13d
  __int64 v7; // r8
  __int64 v8; // r15
  __int64 v10; // r11
  __int64 v11; // rax
  signed int *v12; // rbx
  _DWORD *v13; // rdx
  unsigned __int64 v15; // r12
  signed int *v16; // rax
  signed int v17; // edx
  __int64 v18; // rcx
  unsigned int v19; // ecx
  __int64 v20; // rsi
  _QWORD *v21; // rdi
  _QWORD *v22; // rax
  int v23; // ecx
  __int64 v24; // [rsp-80h] [rbp-80h]
  __int64 v25; // [rsp-70h] [rbp-70h]
  __int64 v26; // [rsp-68h] [rbp-68h]
  __int64 v27; // [rsp-68h] [rbp-68h]
  __int64 v28; // [rsp-60h] [rbp-60h]
  __int64 v29; // [rsp-60h] [rbp-60h]
  unsigned int v30; // [rsp-58h] [rbp-58h]
  __int64 v31; // [rsp-58h] [rbp-58h]
  signed int *v32; // [rsp-50h] [rbp-50h]
  unsigned int v33; // [rsp-50h] [rbp-50h]
  unsigned __int64 v34; // [rsp-40h] [rbp-40h] BYREF

  v3 = 0;
  if ( a3 <= 0 )
    return 0;
  if ( a2 < a3 || a2 % a3 )
    return v3;
  v5 = a2;
  if ( !(_DWORD)a2 )
    return 1;
  v24 = a3;
  v6 = 0;
  v7 = (__int64)a3 >> 2;
  v8 = 4LL * a3;
  v10 = 16 * v7;
  while ( 2 )
  {
    v11 = a1 + 4LL * v6;
    v12 = (signed int *)(v11 + v8);
    if ( v7 )
    {
      v13 = (_DWORD *)(a1 + 4LL * v6);
      while ( *v13 == -1 )
      {
        if ( v13[1] != -1 )
        {
          ++v13;
          goto LABEL_13;
        }
        if ( v13[2] != -1 )
        {
          v13 += 2;
          goto LABEL_13;
        }
        if ( v13[3] != -1 )
        {
          v13 += 3;
          goto LABEL_13;
        }
        v13 += 4;
        if ( (_DWORD *)(v11 + v10) == v13 )
        {
          v18 = v12 - v13;
          goto LABEL_31;
        }
      }
      goto LABEL_13;
    }
    v18 = v24;
    v13 = (_DWORD *)(a1 + 4LL * v6);
LABEL_31:
    if ( v18 == 2 )
    {
LABEL_46:
      if ( *v13 == -1 )
      {
        ++v13;
        goto LABEL_34;
      }
      goto LABEL_13;
    }
    if ( v18 != 3 )
    {
      if ( v18 != 1 )
        goto LABEL_14;
LABEL_34:
      if ( *v13 == -1 )
        goto LABEL_14;
      goto LABEL_13;
    }
    if ( *v13 == -1 )
    {
      ++v13;
      goto LABEL_46;
    }
LABEL_13:
    if ( v12 == v13 )
      goto LABEL_14;
    v25 = a1;
    v26 = v10;
    v28 = v7;
    v30 = v5;
    v32 = (signed int *)(a1 + 4LL * v6);
    sub_B48880((__int64 *)&v34, a3, 0);
    v15 = v34;
    v16 = v32;
    v5 = v30;
    v7 = v28;
    v10 = v26;
    a1 = v25;
    do
    {
      while ( 1 )
      {
        v17 = *v16;
        if ( *v16 == -1 || a3 <= v17 )
          goto LABEL_19;
        if ( (v15 & 1) == 0 )
          break;
        v15 = 2 * ((v15 >> 58 << 57) | ~(-1LL << (v15 >> 58)) & (~(-1LL << (v15 >> 58)) & (v15 >> 1) | (1LL << v17)))
            + 1;
        v34 = v15;
LABEL_19:
        if ( v12 == ++v16 )
          goto LABEL_24;
      }
      ++v16;
      *(_QWORD *)(*(_QWORD *)v15 + 8LL * ((unsigned int)v17 >> 6)) |= 1LL << v17;
      v15 = v34;
    }
    while ( v12 != v16 );
LABEL_24:
    if ( (v15 & 1) != 0 )
    {
      if ( (~(-1LL << (v15 >> 58)) & (v15 >> 1)) != (1LL << (v15 >> 58)) - 1 )
        return 0;
LABEL_14:
      v6 += a3;
      if ( v5 <= v6 )
        return 1;
      continue;
    }
    break;
  }
  v19 = *(_DWORD *)(v15 + 64);
  v20 = v19 >> 6;
  if ( (_DWORD)v20 )
  {
    v21 = *(_QWORD **)v15;
    v22 = *(_QWORD **)v15;
    while ( *v22 == -1 )
    {
      if ( (_QWORD *)(*(_QWORD *)v15 + 8LL * (unsigned int)(v20 - 1) + 8) == ++v22 )
        goto LABEL_48;
    }
  }
  else
  {
LABEL_48:
    v23 = v19 & 0x3F;
    if ( !v23
      || (v21 = *(_QWORD **)v15,
          v20 = (unsigned int)v20,
          *(_QWORD *)(*(_QWORD *)v15 + 8LL * (unsigned int)v20) == (1LL << v23) - 1) )
    {
      if ( v15 )
      {
        if ( *(_QWORD *)v15 != v15 + 16 )
        {
          _libc_free(*(_QWORD *)v15, v20);
          a1 = v25;
          v10 = v26;
          v7 = v28;
          v5 = v30;
        }
        v27 = a1;
        v29 = v10;
        v31 = v7;
        v33 = v5;
        j_j___libc_free_0(v15, 72);
        a1 = v27;
        v10 = v29;
        v7 = v31;
        v5 = v33;
      }
      goto LABEL_14;
    }
  }
  if ( v15 )
  {
    if ( v21 != (_QWORD *)(v15 + 16) )
      _libc_free(v21, v20);
    j_j___libc_free_0(v15, 72);
  }
  return 0;
}
