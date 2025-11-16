// Function: sub_315A010
// Address: 0x315a010
//
__int64 __fastcall sub_315A010(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // r14
  __int64 v12; // r12
  __int64 v13; // r13
  unsigned __int64 v14; // rax
  int v15; // edx
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rax
  int v18; // eax
  unsigned int v19; // esi
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 *v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rdi
  __int64 v25; // r15
  __int64 *v26; // rax
  __int64 v27; // rsi
  char v28; // dl
  __m128i v31; // [rsp+20h] [rbp-50h] BYREF
  char v32; // [rsp+38h] [rbp-38h]

  result = a2[1];
  v8 = a2[2];
LABEL_2:
  while ( 1 )
  {
    v9 = *(_QWORD *)(a3 + 8);
    v10 = *(_QWORD *)(a3 + 16) - v9;
    if ( v8 - result == v10 )
      break;
LABEL_3:
    sub_31599A0(a1, (__int64 *)(v8 - 32), v8, v10, a5, a6);
    v11 = a2[2];
    do
    {
      v12 = *(_QWORD *)(v11 - 32);
      v13 = v12 + 48;
      if ( !*(_BYTE *)(v11 - 8) )
      {
        v14 = *(_QWORD *)(v12 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v14 == v13 )
        {
          v16 = 0;
        }
        else
        {
          if ( !v14 )
            goto LABEL_26;
          v15 = *(unsigned __int8 *)(v14 - 24);
          v16 = v14 - 24;
          if ( (unsigned int)(v15 - 30) >= 0xB )
            v16 = 0;
        }
        *(_QWORD *)(v11 - 24) = v16;
        *(_DWORD *)(v11 - 16) = 0;
        *(_BYTE *)(v11 - 8) = 1;
      }
LABEL_10:
      v17 = *(_QWORD *)(v12 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v17 == v13 )
        goto LABEL_23;
LABEL_11:
      if ( !v17 )
LABEL_26:
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v17 - 24) - 30 <= 0xA )
      {
        v18 = sub_B46E30(v17 - 24);
        v19 = *(_DWORD *)(v11 - 16);
        if ( v19 == v18 )
          goto LABEL_24;
        goto LABEL_14;
      }
LABEL_23:
      while ( 1 )
      {
        v19 = *(_DWORD *)(v11 - 16);
        if ( !v19 )
          break;
LABEL_14:
        v20 = *(_QWORD *)(v11 - 24);
        *(_DWORD *)(v11 - 16) = v19 + 1;
        v21 = sub_B46EC0(v20, v19);
        v24 = *a2;
        v25 = v21;
        if ( *(_BYTE *)(*a2 + 28) )
        {
          v26 = *(__int64 **)(v24 + 8);
          v27 = *(unsigned int *)(v24 + 20);
          v22 = &v26[v27];
          if ( v26 != v22 )
          {
            while ( v25 != *v26 )
            {
              if ( v22 == ++v26 )
                goto LABEL_18;
            }
            goto LABEL_10;
          }
LABEL_18:
          if ( (unsigned int)v27 < *(_DWORD *)(v24 + 16) )
          {
            *(_DWORD *)(v24 + 20) = v27 + 1;
            *v22 = v25;
            ++*(_QWORD *)v24;
LABEL_20:
            v31.m128i_i64[0] = v25;
            v32 = 0;
            sub_3159320((__int64)(a2 + 1), &v31);
            result = a2[1];
            v8 = a2[2];
            goto LABEL_2;
          }
        }
        sub_C8CC70(v24, v25, (__int64)v22, v23, a5, a6);
        if ( v28 )
          goto LABEL_20;
        v17 = *(_QWORD *)(v12 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v17 != v13 )
          goto LABEL_11;
      }
LABEL_24:
      a2[2] -= 32;
      result = a2[1];
      v11 = a2[2];
    }
    while ( v11 != result );
    v8 = a2[1];
  }
  if ( result == v8 )
    return result;
  do
  {
    v10 = *(_QWORD *)v9;
    if ( *(_QWORD *)result != *(_QWORD *)v9 )
      goto LABEL_3;
    v10 = *(unsigned __int8 *)(result + 24);
    if ( (_BYTE)v10 != *(_BYTE *)(v9 + 24) )
      goto LABEL_3;
    if ( (_BYTE)v10 )
    {
      v10 = *(unsigned int *)(v9 + 16);
      if ( *(_DWORD *)(result + 16) != (_DWORD)v10 )
        goto LABEL_3;
    }
    result += 32;
    v9 += 32;
  }
  while ( v8 != result );
  return result;
}
