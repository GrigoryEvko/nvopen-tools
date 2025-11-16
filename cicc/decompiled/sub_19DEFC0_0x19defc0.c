// Function: sub_19DEFC0
// Address: 0x19defc0
//
__int64 __fastcall sub_19DEFC0(__int64 *a1, __int64 a2, __m128i a3, __m128i a4, double a5)
{
  __int64 v5; // r8
  _BYTE *v6; // rcx
  int v7; // eax
  __int64 **v8; // r15
  __int64 v9; // rax
  __int64 *v10; // r13
  __int64 v11; // r9
  _BYTE *i; // rax
  int v13; // r13d
  __int64 v14; // r14
  __int64 *v15; // r13
  __int64 v16; // rax
  int j; // r14d
  __int64 result; // rax
  char v19; // al
  int v20; // edx
  unsigned __int64 v21; // r15
  bool v22; // zf
  __int64 v23; // rax
  _BYTE v24[12]; // [rsp+Ch] [rbp-74h]
  __int64 v25; // [rsp+10h] [rbp-70h]
  int v26; // [rsp+18h] [rbp-68h]
  _BYTE *v27; // [rsp+20h] [rbp-60h] BYREF
  __int64 v28; // [rsp+28h] [rbp-58h]
  _BYTE v29[80]; // [rsp+30h] [rbp-50h] BYREF

  v5 = 0;
  v6 = v29;
  v28 = 0x400000000LL;
  v7 = *(_DWORD *)(a2 + 20);
  v27 = v29;
  v8 = (__int64 **)a1[5];
  v9 = v7 & 0xFFFFFFF;
  v10 = (__int64 *)(a2 + 24 * (1 - v9));
  if ( (__int64 *)a2 != v10 )
  {
    v11 = *v10;
    v5 = 0;
    for ( i = v29; ; i = v27 )
    {
      *(_QWORD *)&i[8 * v5] = v11;
      v10 += 3;
      v5 = (unsigned int)(v28 + 1);
      LODWORD(v28) = v28 + 1;
      if ( (__int64 *)a2 == v10 )
        break;
      v11 = *v10;
      if ( HIDWORD(v28) <= (unsigned int)v5 )
      {
        v25 = *v10;
        sub_16CD150((__int64)&v27, v29, 0, 8, v5, v11);
        v5 = (unsigned int)v28;
        v11 = v25;
      }
    }
    v6 = v27;
    v9 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  }
  v13 = sub_14A26E0(v8, *(_QWORD *)(a2 + 56), *(__int64 **)(a2 - 24 * v9), (__int64)v6, v5);
  if ( v27 != v29 )
    _libc_free((unsigned __int64)v27);
  if ( v13 )
  {
    v14 = (*(_BYTE *)(a2 + 23) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    v15 = (__int64 *)(v14 + 24);
    v16 = sub_16348C0(a2) | 4;
    v26 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    if ( v26 != 1 )
    {
      for ( j = 2; ; j = v20 )
      {
        v21 = v16 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v16 & 4) == 0 )
          break;
        *(_DWORD *)&v24[8] = HIDWORD(v16);
        if ( (v16 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        {
          *(_QWORD *)v24 = (unsigned int)(j - 2);
          v23 = sub_1643D30(0, *v15);
          result = sub_19DEE70(a1, a2, *(unsigned int *)v24, v23, a3, a4, a5);
          if ( result )
            return result;
          v21 = sub_1643D30(*(__int64 *)&v24[4], *v15);
LABEL_22:
          v19 = *(_BYTE *)(v21 + 8);
          if ( ((v19 - 14) & 0xFD) == 0 )
            goto LABEL_18;
          goto LABEL_23;
        }
        result = sub_19DEE70(a1, a2, j - 2, v16 & 0xFFFFFFFFFFFFFFF8LL, a3, a4, a5);
        if ( result )
          return result;
        v19 = *(_BYTE *)(v21 + 8);
        if ( ((v19 - 14) & 0xFD) == 0 )
        {
LABEL_18:
          v15 += 3;
          v20 = j + 1;
          v16 = *(_QWORD *)(v21 + 24) | 4LL;
          if ( v26 == j )
            return 0;
          continue;
        }
LABEL_23:
        v22 = v19 == 13;
        v16 = 0;
        v20 = j + 1;
        if ( v22 )
          v16 = v21;
        v15 += 3;
        if ( v26 == j )
          return 0;
      }
      v21 = sub_1643D30(v16 & 0xFFFFFFFFFFFFFFF8LL, *v15);
      goto LABEL_22;
    }
  }
  return 0;
}
