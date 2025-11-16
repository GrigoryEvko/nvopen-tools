// Function: sub_18DE480
// Address: 0x18de480
//
__int64 __fastcall sub_18DE480(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // r14
  __int64 v5; // rax
  int v6; // eax
  __int64 *v7; // r10
  __int64 v8; // rax
  __int64 *v9; // r8
  __int64 v10; // rax
  __int64 *v11; // r9
  __int64 *v12; // rbx
  __int64 v13; // r12
  char v14; // dl
  __int64 v15; // r13
  __int64 *v16; // rdi
  __int64 *v17; // rax
  __int64 *v18; // rcx
  __int64 result; // rax
  __int64 v20; // rax
  __int64 v22; // r15
  __int64 v23; // r12
  __int64 v24; // r9
  char v25; // di
  __int64 v26; // rax
  unsigned int v27; // esi
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rcx
  __int64 v31; // rdx
  __int64 v32; // [rsp+8h] [rbp-98h]
  unsigned __int64 v33; // [rsp+10h] [rbp-90h]
  unsigned __int8 v35; // [rsp+18h] [rbp-88h]
  __int64 v36; // [rsp+20h] [rbp-80h] BYREF
  __int64 *v37; // [rsp+28h] [rbp-78h]
  __int64 *v38; // [rsp+30h] [rbp-70h]
  __int64 v39; // [rsp+38h] [rbp-68h]
  int v40; // [rsp+40h] [rbp-60h]
  _BYTE v41[88]; // [rsp+48h] [rbp-58h] BYREF

  v3 = (__int64 *)a2;
  v5 = sub_15F2050(a2);
  v33 = sub_1632FA0(v5);
  if ( *(_BYTE *)(a3 + 16) == 77 && *(_QWORD *)(a3 + 40) == *(_QWORD *)(a2 + 40) )
  {
    v32 = 8LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 0 )
    {
      v20 = a3;
      v22 = 0;
      v23 = v20;
      while ( 1 )
      {
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v24 = *(_QWORD *)(a2 - 8);
        else
          v24 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        v25 = *(_BYTE *)(v23 + 23) & 0x40;
        v26 = 0x17FFFFFFE8LL;
        v27 = *(_DWORD *)(v23 + 20) & 0xFFFFFFF;
        if ( v27 )
        {
          v28 = 24LL * *(unsigned int *)(v23 + 56) + 8;
          v29 = 0;
          do
          {
            v30 = v23 - 24LL * v27;
            if ( v25 )
              v30 = *(_QWORD *)(v23 - 8);
            if ( *(_QWORD *)(v22 + v24 + 24LL * *(unsigned int *)(a2 + 56) + 8) == *(_QWORD *)(v30 + v28) )
            {
              v26 = 24 * v29;
              goto LABEL_36;
            }
            ++v29;
            v28 += 8;
          }
          while ( v27 != (_DWORD)v29 );
          v26 = 0x17FFFFFFE8LL;
        }
LABEL_36:
        v31 = v25 ? *(_QWORD *)(v23 - 8) : v23 - 24LL * v27;
        result = sub_18DDD00(a1, *(_QWORD *)(v24 + 3 * v22), *(_QWORD *)(v31 + v26), v33);
        if ( (_BYTE)result )
          return result;
        v22 += 8;
        if ( v22 == v32 )
          return 0;
      }
    }
    return 0;
  }
  v6 = *(_DWORD *)(a2 + 20);
  v7 = (__int64 *)v41;
  v36 = 0;
  v37 = (__int64 *)v41;
  v38 = (__int64 *)v41;
  v39 = 4;
  v8 = 3LL * (v6 & 0xFFFFFFF);
  v40 = 0;
  v9 = (__int64 *)(a2 - v8 * 8);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v9 = *(__int64 **)(a2 - 8);
    v3 = &v9[v8];
  }
  if ( v3 == v9 )
    return 0;
  v10 = a3;
  v11 = (__int64 *)v41;
  v12 = v9;
  v13 = v10;
  do
  {
    v15 = *v12;
    if ( v11 != v7 )
      goto LABEL_6;
    v16 = &v11[HIDWORD(v39)];
    if ( v16 != v11 )
    {
      v17 = v11;
      v18 = 0;
      while ( v15 != *v17 )
      {
        if ( *v17 == -2 )
          v18 = v17;
        if ( v16 == ++v17 )
        {
          if ( !v18 )
            goto LABEL_21;
          *v18 = v15;
          --v40;
          ++v36;
          goto LABEL_17;
        }
      }
      goto LABEL_7;
    }
LABEL_21:
    if ( HIDWORD(v39) < (unsigned int)v39 )
    {
      ++HIDWORD(v39);
      *v16 = v15;
      ++v36;
    }
    else
    {
LABEL_6:
      sub_16CCBA0((__int64)&v36, *v12);
      v11 = v38;
      v7 = v37;
      if ( !v14 )
        goto LABEL_7;
    }
LABEL_17:
    result = sub_18DDD00(a1, v15, v13, v33);
    v11 = v38;
    v7 = v37;
    if ( (_BYTE)result )
      goto LABEL_18;
LABEL_7:
    v12 += 3;
  }
  while ( v3 != v12 );
  result = 0;
LABEL_18:
  if ( v7 != v11 )
  {
    v35 = result;
    _libc_free((unsigned __int64)v11);
    return v35;
  }
  return result;
}
