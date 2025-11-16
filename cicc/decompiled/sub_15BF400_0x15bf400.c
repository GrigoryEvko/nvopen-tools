// Function: sub_15BF400
// Address: 0x15bf400
//
__int64 __fastcall sub_15BF400(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v4; // r12d
  int v6; // esi
  __int64 *v7; // rcx
  __int64 v8; // rdi
  int v9; // eax
  __int64 v10; // r14
  __int64 v11; // rdx
  char v12; // al
  int v13; // eax
  unsigned int v14; // r12d
  unsigned int v15; // edx
  __int64 *v16; // rsi
  int v17; // r8d
  __int64 *v18; // r9
  int v19; // eax
  __int64 v20; // [rsp+8h] [rbp-88h] BYREF
  int v21; // [rsp+1Ch] [rbp-74h] BYREF
  __int64 v22; // [rsp+20h] [rbp-70h] BYREF
  __int64 v23; // [rsp+28h] [rbp-68h] BYREF
  __int64 *v24; // [rsp+30h] [rbp-60h] BYREF
  __int64 v25; // [rsp+38h] [rbp-58h] BYREF
  __m128i v26; // [rsp+40h] [rbp-50h]
  char v27; // [rsp+50h] [rbp-40h]
  __int64 v28; // [rsp+58h] [rbp-38h]
  char v29; // [rsp+60h] [rbp-30h]

  v20 = a1;
  if ( (_DWORD)a2 )
  {
    result = a1;
    if ( (_DWORD)a2 == 1 )
    {
      sub_1621390(a1, a2);
      return v20;
    }
    return result;
  }
  v4 = *(_DWORD *)(a3 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a3;
    goto LABEL_7;
  }
  v10 = *(_QWORD *)(a3 + 8);
  v11 = *(unsigned int *)(a1 + 8);
  v24 = *(__int64 **)(a1 - 8 * v11);
  v25 = *(_QWORD *)(a1 + 8 * (1 - v11));
  v27 = *(_BYTE *)(a1 + 40);
  v12 = *(_BYTE *)(a1 + 56);
  if ( v27 )
  {
    v29 = *(_BYTE *)(a1 + 56);
    v26 = _mm_loadu_si128((const __m128i *)(a1 + 24));
    if ( v12 )
    {
      v28 = *(_QWORD *)(a1 + 48);
      v23 = v28;
    }
    else
    {
      v23 = 0;
    }
    v22 = v26.m128i_i64[1];
    v13 = v26.m128i_i32[0];
  }
  else
  {
    v29 = *(_BYTE *)(a1 + 56);
    if ( v12 )
    {
      v28 = *(_QWORD *)(a1 + 48);
      v23 = v28;
    }
    else
    {
      v23 = 0;
    }
    v22 = 0;
    v13 = 0;
  }
  v21 = v13;
  v14 = v4 - 1;
  v8 = v20;
  v15 = v14 & sub_15B5960((__int64 *)&v24, &v25, &v21, &v22, &v23);
  v16 = (__int64 *)(v10 + 8LL * v15);
  result = *v16;
  if ( v20 != *v16 )
  {
    v17 = 1;
    v7 = 0;
    if ( result == -8 )
    {
LABEL_25:
      v19 = *(_DWORD *)(a3 + 16);
      v4 = *(_DWORD *)(a3 + 24);
      if ( !v7 )
        v7 = v16;
      ++*(_QWORD *)a3;
      v9 = v19 + 1;
      if ( 4 * v9 < 3 * v4 )
      {
        if ( v4 - (v9 + *(_DWORD *)(a3 + 20)) > v4 >> 3 )
          goto LABEL_9;
        v6 = v4;
LABEL_8:
        sub_15BF140(a3, v6);
        sub_15B81B0(a3, &v20, &v24);
        v7 = v24;
        v8 = v20;
        v9 = *(_DWORD *)(a3 + 16) + 1;
LABEL_9:
        *(_DWORD *)(a3 + 16) = v9;
        if ( *v7 != -8 )
          --*(_DWORD *)(a3 + 20);
        *v7 = v8;
        return v20;
      }
LABEL_7:
      v6 = 2 * v4;
      goto LABEL_8;
    }
    while ( 1 )
    {
      if ( result != -16 || v7 )
        v16 = v7;
      v15 = v14 & (v17 + v15);
      v18 = (__int64 *)(v10 + 8LL * v15);
      result = *v18;
      if ( *v18 == v20 )
        break;
      ++v17;
      v7 = v16;
      v16 = (__int64 *)(v10 + 8LL * v15);
      if ( result == -8 )
        goto LABEL_25;
    }
  }
  return result;
}
