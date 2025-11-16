// Function: sub_3372B70
// Address: 0x3372b70
//
__int64 __fastcall sub_3372B70(__int64 a1, __int64 a2)
{
  _DWORD *v3; // r10
  _WORD *v4; // r11
  __int64 v5; // r9
  unsigned int v6; // ebx
  int v7; // eax
  __int64 v8; // rax
  __int64 v9; // r8
  char v10; // al
  __int64 v11; // rdx
  unsigned int v12; // r10d
  __int64 v13; // r14
  char v14; // r11
  __int64 v15; // rsi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // r8
  const __m128i *v18; // r13
  int v19; // ecx
  unsigned __int64 v20; // rcx
  __m128i *v21; // rdx
  char *v23; // r13
  __int64 v24; // [rsp+0h] [rbp-80h]
  unsigned int v25; // [rsp+8h] [rbp-78h]
  char v26; // [rsp+Fh] [rbp-71h]
  _DWORD *v27; // [rsp+10h] [rbp-70h]
  const void *v28; // [rsp+18h] [rbp-68h]
  _WORD *v29; // [rsp+20h] [rbp-60h]
  _DWORD *v30; // [rsp+28h] [rbp-58h]
  int v31; // [rsp+30h] [rbp-50h] BYREF
  __int64 v32; // [rsp+38h] [rbp-48h]
  char v33; // [rsp+40h] [rbp-40h]

  *(_QWORD *)a1 = a1 + 16;
  v28 = (const void *)(a1 + 16);
  *(_QWORD *)(a1 + 8) = 0x400000000LL;
  v3 = *(_DWORD **)(a2 + 144);
  v4 = *(_WORD **)(a2 + 80);
  v27 = &v3[*(unsigned int *)(a2 + 152)];
  if ( v3 != v27 )
  {
    v5 = a2;
    v6 = 0;
    while ( 1 )
    {
      v7 = (unsigned __int16)*v4;
      if ( (unsigned __int16)v7 <= 1u || (unsigned __int16)(*v4 - 504) <= 7u )
        BUG();
      v8 = 16LL * (v7 - 1);
      v9 = *(_QWORD *)&byte_444C4A0[v8];
      v10 = byte_444C4A0[v8 + 8];
      if ( v6 + *v3 != v6 )
        break;
LABEL_10:
      ++v4;
      if ( v27 == ++v3 )
        return a1;
    }
    v11 = *(unsigned int *)(a1 + 8);
    v30 = v3;
    v12 = v6 + *v3;
    v13 = v9;
    v29 = v4;
    v14 = v10;
    while ( 1 )
    {
      v15 = *(_QWORD *)(v5 + 112);
      v16 = *(unsigned int *)(a1 + 12);
      v17 = v11 + 1;
      v32 = v13;
      v18 = (const __m128i *)&v31;
      v19 = *(_DWORD *)(v15 + 4LL * v6);
      v33 = v14;
      v31 = v19;
      v20 = *(_QWORD *)a1;
      if ( v11 + 1 > v16 )
      {
        if ( v20 > (unsigned __int64)&v31 )
        {
          v24 = v5;
          v25 = v12;
          v26 = v14;
LABEL_16:
          sub_C8D5F0(a1, v28, v17, 0x18u, v17, v5);
          v20 = *(_QWORD *)a1;
          v11 = *(unsigned int *)(a1 + 8);
          v14 = v26;
          v12 = v25;
          v5 = v24;
          goto LABEL_8;
        }
        v24 = v5;
        v25 = v12;
        v26 = v14;
        if ( (unsigned __int64)&v31 >= v20 + 24 * v11 )
          goto LABEL_16;
        v23 = (char *)&v31 - v20;
        sub_C8D5F0(a1, v28, v17, 0x18u, v17, v5);
        v20 = *(_QWORD *)a1;
        v11 = *(unsigned int *)(a1 + 8);
        v5 = v24;
        v12 = v25;
        v14 = v26;
        v18 = (const __m128i *)&v23[*(_QWORD *)a1];
      }
LABEL_8:
      ++v6;
      v21 = (__m128i *)(v20 + 24 * v11);
      *v21 = _mm_loadu_si128(v18);
      v21[1].m128i_i64[0] = v18[1].m128i_i64[0];
      v11 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
      *(_DWORD *)(a1 + 8) = v11;
      if ( v12 == v6 )
      {
        v3 = v30;
        v4 = v29;
        goto LABEL_10;
      }
    }
  }
  return a1;
}
