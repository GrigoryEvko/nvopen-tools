// Function: sub_2B1EE10
// Address: 0x2b1ee10
//
char __fastcall sub_2B1EE10(__int64 *a1, _BYTE *a2, unsigned int *a3)
{
  __int64 v6; // rsi
  char v7; // cl
  __int64 v8; // rdx
  __int64 v9; // r8
  int v10; // eax
  unsigned int v11; // edx
  __int64 v12; // rdi
  _BYTE *v13; // r9
  __int64 v14; // rax
  char result; // al
  unsigned __int64 v16; // rax
  char v17; // al
  unsigned int v18; // ebx
  unsigned int v19; // edx
  __int64 v20; // rdi
  unsigned __int64 v21; // rax
  int v22; // edi
  int v23; // r10d
  char v24; // [rsp-A0h] [rbp-A0h]
  unsigned __int64 v25; // [rsp-98h] [rbp-98h] BYREF
  unsigned int v26; // [rsp-90h] [rbp-90h]
  __m128i v27; // [rsp-88h] [rbp-88h] BYREF
  __int64 v28; // [rsp-78h] [rbp-78h]
  __int64 v29; // [rsp-70h] [rbp-70h]
  __int64 v30; // [rsp-68h] [rbp-68h]
  __int64 v31; // [rsp-60h] [rbp-60h]
  __int64 v32; // [rsp-58h] [rbp-58h]
  __int64 v33; // [rsp-50h] [rbp-50h]
  __int16 v34; // [rsp-48h] [rbp-48h]

  if ( *a2 == 13 )
    return 1;
  v6 = *a1;
  v7 = *(_BYTE *)(*a1 + 88) & 1;
  if ( v7 )
  {
    v9 = v6 + 96;
    v10 = 3;
  }
  else
  {
    v8 = *(unsigned int *)(v6 + 104);
    v9 = *(_QWORD *)(v6 + 96);
    if ( !(_DWORD)v8 )
      goto LABEL_18;
    v10 = v8 - 1;
  }
  v11 = v10 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v12 = v9 + 72LL * v11;
  v13 = *(_BYTE **)v12;
  if ( a2 == *(_BYTE **)v12 )
    goto LABEL_6;
  v22 = 1;
  while ( v13 != (_BYTE *)-4096LL )
  {
    v23 = v22 + 1;
    v11 = v10 & (v22 + v11);
    v12 = v9 + 72LL * v11;
    v13 = *(_BYTE **)v12;
    if ( a2 == *(_BYTE **)v12 )
      goto LABEL_6;
    v22 = v23;
  }
  if ( v7 )
  {
    v20 = 288;
    goto LABEL_19;
  }
  v8 = *(unsigned int *)(v6 + 104);
LABEL_18:
  v20 = 72 * v8;
LABEL_19:
  v12 = v9 + v20;
LABEL_6:
  v14 = 288;
  if ( !v7 )
    v14 = 72LL * *(unsigned int *)(v6 + 104);
  if ( v12 == v9 + v14 || (result = 0, *(_DWORD *)(v12 + 16) <= 1u) )
  {
    v16 = *(_QWORD *)(v6 + 3344);
    v34 = 257;
    v27 = (__m128i)v16;
    v28 = 0;
    v29 = 0;
    v30 = 0;
    v31 = 0;
    v32 = 0;
    v33 = 0;
    v17 = sub_9AC470((__int64)a2, &v27, 0);
    if ( *(_BYTE *)a1[1] && v17 )
      return sub_2B17B70((_DWORD **)a1, a2, a3);
    v18 = *a3;
    v19 = *(_DWORD *)a1[2];
    if ( v19 <= *a3 )
      return sub_2B17B70((_DWORD **)a1, a2, a3);
    v26 = *(_DWORD *)a1[2];
    if ( v19 > 0x40 )
    {
      sub_C43690((__int64)&v25, 0, 0);
      v19 = v26;
      if ( v18 == v26 )
      {
LABEL_25:
        v21 = *(_QWORD *)(*a1 + 3344);
        v28 = 0;
        v27 = (__m128i)v21;
        v29 = 0;
        v30 = 0;
        v31 = 0;
        v32 = 0;
        v33 = 0;
        v34 = 257;
        result = sub_9AC230((__int64)a2, (__int64)&v25, &v27, 0);
        if ( result )
        {
          if ( v26 > 0x40 && v25 )
          {
            v24 = result;
            j_j___libc_free_0_0(v25);
            return v24;
          }
          return result;
        }
        if ( v26 > 0x40 )
        {
          if ( v25 )
            j_j___libc_free_0_0(v25);
        }
        return sub_2B17B70((_DWORD **)a1, a2, a3);
      }
    }
    else
    {
      v25 = 0;
    }
    if ( v18 > 0x3F || v19 > 0x40 )
      sub_C43C90(&v25, v18, v19);
    else
      v25 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v18 + 64 - (unsigned __int8)v19) << v18;
    goto LABEL_25;
  }
  return result;
}
