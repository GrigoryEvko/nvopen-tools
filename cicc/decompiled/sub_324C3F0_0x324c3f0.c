// Function: sub_324C3F0
// Address: 0x324c3f0
//
__int64 __fastcall sub_324C3F0(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  unsigned int v6; // esi
  __int64 v7; // r8
  unsigned int v8; // edi
  __int64 result; // rax
  unsigned __int8 *v10; // rcx
  __int64 v11; // rsi
  int v12; // eax
  int v13; // esi
  __int64 v14; // r8
  int v15; // ecx
  unsigned __int8 **v16; // rdx
  unsigned __int8 *v17; // rdi
  int v18; // r11d
  int v19; // eax
  int v20; // eax
  __int64 v21; // rdi
  unsigned __int8 **v22; // r8
  unsigned int v23; // r14d
  int v24; // r9d
  unsigned __int8 *v25; // rsi
  int v26; // r10d
  unsigned __int8 **v27; // r9
  unsigned __int8 *v28; // [rsp+0h] [rbp-70h] BYREF
  __int64 v29; // [rsp+8h] [rbp-68h] BYREF
  char v30[96]; // [rsp+10h] [rbp-60h] BYREF

  if ( (unsigned __int8)sub_3247C00((_QWORD *)a1, a2) )
  {
    v11 = *(_QWORD *)(a1 + 216);
    v28 = a2;
    v29 = a3;
    return sub_324BEF0((__int64)v30, v11 + 464, (__int64 *)&v28, &v29);
  }
  v6 = *(_DWORD *)(a1 + 256);
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 232);
    goto LABEL_7;
  }
  v7 = *(_QWORD *)(a1 + 240);
  v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = v7 + 16LL * v8;
  v10 = *(unsigned __int8 **)result;
  if ( a2 == *(unsigned __int8 **)result )
    return result;
  v18 = 1;
  v16 = 0;
  while ( v10 != (unsigned __int8 *)-4096LL )
  {
    if ( v10 != (unsigned __int8 *)-8192LL || v16 )
      result = (__int64)v16;
    v8 = (v6 - 1) & (v18 + v8);
    v10 = *(unsigned __int8 **)(v7 + 16LL * v8);
    if ( a2 == v10 )
      return result;
    ++v18;
    v16 = (unsigned __int8 **)result;
    result = v7 + 16LL * v8;
  }
  if ( !v16 )
    v16 = (unsigned __int8 **)result;
  v19 = *(_DWORD *)(a1 + 248);
  ++*(_QWORD *)(a1 + 232);
  v15 = v19 + 1;
  if ( 4 * (v19 + 1) >= 3 * v6 )
  {
LABEL_7:
    sub_324BD10(a1 + 232, 2 * v6);
    v12 = *(_DWORD *)(a1 + 256);
    if ( v12 )
    {
      v13 = v12 - 1;
      v14 = *(_QWORD *)(a1 + 240);
      result = (v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = *(_DWORD *)(a1 + 248) + 1;
      v16 = (unsigned __int8 **)(v14 + 16 * result);
      v17 = *v16;
      if ( a2 != *v16 )
      {
        v26 = 1;
        v27 = 0;
        while ( v17 != (unsigned __int8 *)-4096LL )
        {
          if ( v17 == (unsigned __int8 *)-8192LL && !v27 )
            v27 = v16;
          result = v13 & (unsigned int)(v26 + result);
          v16 = (unsigned __int8 **)(v14 + 16LL * (unsigned int)result);
          v17 = *v16;
          if ( a2 == *v16 )
            goto LABEL_9;
          ++v26;
        }
        if ( v27 )
          v16 = v27;
      }
      goto LABEL_9;
    }
    goto LABEL_44;
  }
  result = v6 - *(_DWORD *)(a1 + 252) - v15;
  if ( (unsigned int)result <= v6 >> 3 )
  {
    sub_324BD10(a1 + 232, v6);
    v20 = *(_DWORD *)(a1 + 256);
    if ( v20 )
    {
      result = (unsigned int)(v20 - 1);
      v21 = *(_QWORD *)(a1 + 240);
      v22 = 0;
      v23 = result & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v24 = 1;
      v15 = *(_DWORD *)(a1 + 248) + 1;
      v16 = (unsigned __int8 **)(v21 + 16LL * v23);
      v25 = *v16;
      if ( a2 != *v16 )
      {
        while ( v25 != (unsigned __int8 *)-4096LL )
        {
          if ( !v22 && v25 == (unsigned __int8 *)-8192LL )
            v22 = v16;
          v23 = result & (v24 + v23);
          v16 = (unsigned __int8 **)(v21 + 16LL * v23);
          v25 = *v16;
          if ( a2 == *v16 )
            goto LABEL_9;
          ++v24;
        }
        if ( v22 )
          v16 = v22;
      }
      goto LABEL_9;
    }
LABEL_44:
    ++*(_DWORD *)(a1 + 248);
    BUG();
  }
LABEL_9:
  *(_DWORD *)(a1 + 248) = v15;
  if ( *v16 != (unsigned __int8 *)-4096LL )
    --*(_DWORD *)(a1 + 252);
  *v16 = a2;
  v16[1] = (unsigned __int8 *)a3;
  return result;
}
