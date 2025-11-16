// Function: sub_1095810
// Address: 0x1095810
//
__int64 __fastcall sub_1095810(__int64 a1, unsigned int a2, __int64 a3, unsigned __int64 a4, __int64 a5)
{
  __int64 v5; // r14
  unsigned int v7; // ebx
  unsigned int v9; // r15d
  unsigned __int64 v10; // rdx
  __int64 v11; // rsi
  _BYTE *v12; // r8
  int v13; // ecx
  unsigned int v14; // eax
  unsigned int v15; // esi
  __int64 v16; // rax
  __int64 v17; // r10
  __int64 v18; // rdx
  __m128i *v19; // rax
  __int64 v20; // rcx
  _QWORD *v21; // rdi
  __int64 v22; // rbx
  _QWORD *v23; // [rsp+10h] [rbp-50h] BYREF
  int v24; // [rsp+18h] [rbp-48h]
  _QWORD v25[8]; // [rsp+20h] [rbp-40h] BYREF

  v5 = a1 + 16;
  if ( a2 == 10 )
  {
    *(_QWORD *)a1 = v5;
    *(_DWORD *)(a1 + 16) = 1768121700;
    *(_WORD *)(a1 + 20) = 24941;
    *(_BYTE *)(a1 + 22) = 108;
    *(_QWORD *)(a1 + 8) = 7;
    *(_BYTE *)(a1 + 23) = 0;
    return a1;
  }
  v7 = a2;
  if ( a2 <= 0xA )
  {
    if ( a2 == 2 )
    {
      *(_QWORD *)a1 = v5;
      strcpy((char *)(a1 + 16), "binary");
      *(_QWORD *)(a1 + 8) = 6;
      return a1;
    }
    if ( a2 == 8 )
    {
      *(_QWORD *)a1 = v5;
      *(_DWORD *)(a1 + 16) = 1635017583;
      *(_BYTE *)(a1 + 20) = 108;
      *(_QWORD *)(a1 + 8) = 5;
      *(_BYTE *)(a1 + 21) = 0;
      return a1;
    }
    v23 = v25;
    sub_2240A50(&v23, 1, 0, a4, a5);
    v12 = v23;
    goto LABEL_24;
  }
  if ( a2 == 16 )
  {
    *(_QWORD *)a1 = v5;
    strcpy((char *)(a1 + 16), "hexadecimal");
    *(_QWORD *)(a1 + 8) = 11;
    return a1;
  }
  if ( a2 <= 0x63 )
  {
    v23 = v25;
    sub_2240A50(&v23, 2, 0, a4, a5);
    v12 = v23;
LABEL_30:
    v22 = 2 * v7;
    v12[1] = a00010203040506[(unsigned int)(v22 + 1)];
    *v12 = a00010203040506[v22];
    goto LABEL_25;
  }
  if ( a2 <= 0x3E7 )
  {
    v11 = 3;
    v9 = v7;
  }
  else
  {
    v9 = a2;
    v10 = a2;
    if ( a2 <= 0x270F )
    {
      v11 = 4;
    }
    else
    {
      LODWORD(v11) = 1;
      while ( 1 )
      {
        a4 = v10;
        a5 = (unsigned int)v11;
        v11 = (unsigned int)(v11 + 4);
        v10 /= 0x2710u;
        if ( a4 <= 0x1869F )
          break;
        if ( (unsigned int)v10 <= 0x63 )
        {
          v23 = v25;
          v11 = (unsigned int)(a5 + 5);
          goto LABEL_20;
        }
        if ( (unsigned int)v10 <= 0x3E7 )
        {
          v11 = (unsigned int)(a5 + 6);
          break;
        }
        if ( (unsigned int)v10 <= 0x270F )
        {
          v11 = (unsigned int)(a5 + 7);
          break;
        }
      }
    }
  }
  v23 = v25;
LABEL_20:
  sub_2240A50(&v23, v11, 0, a4, a5);
  v12 = v23;
  v13 = v24 - 1;
  while ( 1 )
  {
    v14 = v7 - 100 * (v9 / 0x64);
    v15 = v7;
    v7 = v9 / 0x64;
    v16 = 2 * v14;
    v17 = (unsigned int)(v16 + 1);
    LOBYTE(v16) = a00010203040506[v16];
    v12[v13] = a00010203040506[v17];
    v18 = (unsigned int)(v13 - 1);
    v13 -= 2;
    v12[v18] = v16;
    if ( v15 <= 0x270F )
      break;
    v9 /= 0x64u;
  }
  if ( v15 > 0x3E7 )
    goto LABEL_30;
LABEL_24:
  *v12 = v7 + 48;
LABEL_25:
  v19 = (__m128i *)sub_2241130(&v23, 0, 0, "base-", 5);
  *(_QWORD *)a1 = v5;
  if ( (__m128i *)v19->m128i_i64[0] == &v19[1] )
  {
    *(__m128i *)(a1 + 16) = _mm_loadu_si128(v19 + 1);
  }
  else
  {
    *(_QWORD *)a1 = v19->m128i_i64[0];
    *(_QWORD *)(a1 + 16) = v19[1].m128i_i64[0];
  }
  v20 = v19->m128i_i64[1];
  v19->m128i_i64[0] = (__int64)v19[1].m128i_i64;
  v21 = v23;
  v19->m128i_i64[1] = 0;
  *(_QWORD *)(a1 + 8) = v20;
  v19[1].m128i_i8[0] = 0;
  if ( v21 != v25 )
    j_j___libc_free_0(v21, v25[0] + 1LL);
  return a1;
}
