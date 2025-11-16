// Function: sub_1622EA0
// Address: 0x1622ea0
//
__int64 __fastcall sub_1622EA0(__int64 a1, __int64 a2, __int64 *a3, const __m128i *a4)
{
  char v8; // dl
  __int64 v9; // rsi
  int v10; // edx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  int v14; // r11d
  unsigned int v15; // eax
  __int64 *v16; // rdi
  __int64 v17; // r10
  __int64 v19; // r15
  unsigned int v20; // eax
  __int64 *v21; // r9
  int v22; // ecx
  unsigned int v23; // esi
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  int v27; // r15d
  unsigned int v28; // esi
  __int64 *v29; // [rsp+18h] [rbp-38h] BYREF

  v8 = *(_BYTE *)(a2 + 8);
  v9 = *(_QWORD *)a2;
  v10 = v8 & 1;
  if ( v10 )
  {
    v11 = *a3;
    v12 = a2 + 16;
    v13 = 96;
    v14 = 3;
    v15 = ((unsigned __int8)((unsigned int)*a3 >> 9) ^ (unsigned __int8)((unsigned int)*a3 >> 4)) & 3;
    v16 = (__int64 *)(a2
                    + 16
                    + 24LL
                    * (((unsigned __int8)((unsigned int)*a3 >> 9) ^ (unsigned __int8)((unsigned int)*a3 >> 4)) & 3));
    v17 = *v16;
    if ( *a3 == *v16 )
    {
LABEL_3:
      *(_QWORD *)a1 = a2;
      *(_QWORD *)(a1 + 8) = v9;
      *(_QWORD *)(a1 + 16) = v16;
      *(_QWORD *)(a1 + 24) = v13 + v12;
      *(_BYTE *)(a1 + 32) = 0;
      return a1;
    }
    goto LABEL_18;
  }
  v19 = *(unsigned int *)(a2 + 24);
  if ( (_DWORD)v19 )
  {
    v11 = *a3;
    v14 = v19 - 1;
    v12 = *(_QWORD *)(a2 + 16);
    v15 = (v19 - 1) & (((unsigned int)*a3 >> 4) ^ ((unsigned int)*a3 >> 9));
    v16 = (__int64 *)(v12 + 24LL * v15);
    v17 = *v16;
    if ( *a3 == *v16 )
    {
LABEL_7:
      v13 = 24 * v19;
      goto LABEL_3;
    }
LABEL_18:
    v27 = 1;
    v21 = 0;
    while ( 1 )
    {
      if ( v17 == -4 )
      {
        v20 = *(_DWORD *)(a2 + 8);
        if ( !v21 )
          v21 = v16;
        *(_QWORD *)a2 = v9 + 1;
        v22 = (v20 >> 1) + 1;
        if ( (_BYTE)v10 )
        {
          v23 = 12;
          LODWORD(v19) = 4;
          goto LABEL_10;
        }
        LODWORD(v19) = *(_DWORD *)(a2 + 24);
        goto LABEL_9;
      }
      if ( !v21 && v17 == -8 )
        v21 = v16;
      v15 = v14 & (v27 + v15);
      v16 = (__int64 *)(v12 + 24LL * v15);
      v17 = *v16;
      if ( *v16 == v11 )
        break;
      ++v27;
    }
    if ( !(_BYTE)v10 )
    {
      v19 = *(unsigned int *)(a2 + 24);
      goto LABEL_7;
    }
    v13 = 96;
    goto LABEL_3;
  }
  v20 = *(_DWORD *)(a2 + 8);
  v21 = 0;
  *(_QWORD *)a2 = v9 + 1;
  v22 = (v20 >> 1) + 1;
LABEL_9:
  v23 = 3 * v19;
LABEL_10:
  if ( v23 <= 4 * v22 )
  {
    v28 = 2 * v19;
LABEL_25:
    sub_1622AB0(a2, v28);
    sub_1621140(a2, a3, &v29);
    v21 = v29;
    v20 = *(_DWORD *)(a2 + 8);
    goto LABEL_12;
  }
  if ( (int)v19 - *(_DWORD *)(a2 + 12) - v22 <= (unsigned int)v19 >> 3 )
  {
    v28 = v19;
    goto LABEL_25;
  }
LABEL_12:
  *(_DWORD *)(a2 + 8) = (2 * (v20 >> 1) + 2) | v20 & 1;
  if ( *v21 != -4 )
    --*(_DWORD *)(a2 + 12);
  *v21 = *a3;
  *(__m128i *)(v21 + 1) = _mm_loadu_si128(a4);
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v24 = a2 + 16;
    v25 = 96;
  }
  else
  {
    v24 = *(_QWORD *)(a2 + 16);
    v25 = 24LL * *(unsigned int *)(a2 + 24);
  }
  v26 = *(_QWORD *)a2;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v21;
  *(_QWORD *)(a1 + 8) = v26;
  *(_QWORD *)(a1 + 24) = v25 + v24;
  *(_BYTE *)(a1 + 32) = 1;
  return a1;
}
