// Function: sub_F0D210
// Address: 0xf0d210
//
unsigned __int8 *__fastcall sub_F0D210(__int64 a1, unsigned __int8 *a2)
{
  unsigned int v2; // r10d
  __int64 v3; // rbx
  unsigned __int8 *v4; // r14
  int v5; // eax
  int v6; // r15d
  unsigned int v7; // eax
  unsigned __int8 *v8; // r15
  unsigned __int8 *v9; // rax
  unsigned int v10; // r11d
  unsigned int v11; // r10d
  unsigned __int8 *result; // rax
  unsigned __int8 *v13; // rax
  unsigned int v14; // eax
  unsigned int v15; // r11d
  unsigned int v16; // [rsp+8h] [rbp-58h]
  unsigned int v17; // [rsp+8h] [rbp-58h]
  unsigned int v18; // [rsp+8h] [rbp-58h]
  unsigned int v19; // [rsp+Ch] [rbp-54h]
  unsigned int v20; // [rsp+Ch] [rbp-54h]
  unsigned int v21; // [rsp+Ch] [rbp-54h]
  unsigned int v22; // [rsp+Ch] [rbp-54h]
  unsigned int v23; // [rsp+Ch] [rbp-54h]
  unsigned int v24; // [rsp+Ch] [rbp-54h]
  unsigned int v25; // [rsp+Ch] [rbp-54h]
  __int64 v26; // [rsp+10h] [rbp-50h] BYREF
  __int64 v27; // [rsp+18h] [rbp-48h] BYREF
  __int64 v28; // [rsp+20h] [rbp-40h] BYREF
  __int64 v29[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = *((_QWORD *)a2 - 8);
  v4 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
  v5 = *v4;
  if ( (unsigned __int8)(*(_BYTE *)v3 - 42) > 0x11u )
  {
    if ( (unsigned __int8)v5 <= 0x1Cu || (unsigned int)(v5 - 42) > 0x11 )
      return 0;
    v11 = sub_F09C10((unsigned int)*a2 - 29, *((unsigned __int8 **)a2 - 4), &v28, v29, 0);
LABEL_13:
    if ( *(_BYTE *)v3 > 0x15u )
    {
      v22 = v11;
      v13 = sub_AD93D0(v11, *(_QWORD *)(v3 + 8), 0, 0);
      if ( v13 )
        return sub_F0CA40(
                 a1,
                 a2,
                 (const __m128i *)(a1 + 96),
                 *(__int64 **)(a1 + 32),
                 v22,
                 v3,
                 (__int64)v13,
                 v28,
                 v29[0]);
    }
    return 0;
  }
  v6 = *a2 - 29;
  if ( (unsigned __int8)v5 <= 0x1Cu || (unsigned int)(v5 - 42) > 0x11 )
  {
    v19 = v2;
    v7 = sub_F09C10(v6, *((unsigned __int8 **)a2 - 8), &v26, &v27, 0);
    if ( *v4 > 0x15u )
    {
      v16 = v19;
      v8 = 0;
      v20 = v7;
      v9 = sub_AD93D0(v7, *((_QWORD *)v4 + 1), 0, 0);
      v10 = v20;
      v11 = v16;
      if ( v9 )
      {
LABEL_6:
        v21 = v11;
        result = sub_F0CA40(
                   a1,
                   a2,
                   (const __m128i *)(a1 + 96),
                   *(__int64 **)(a1 + 32),
                   v10,
                   v26,
                   v27,
                   (__int64)v4,
                   (__int64)v9);
        if ( result )
          return result;
        v11 = v21;
        if ( !v8 )
          return 0;
        goto LABEL_13;
      }
    }
    return 0;
  }
  v23 = sub_F09C10(v6, *((unsigned __int8 **)a2 - 8), &v26, &v27, *((_BYTE **)a2 - 4));
  v14 = sub_F09C10(v6, v4, &v28, v29, (_BYTE *)v3);
  v15 = v23;
  v11 = v14;
  if ( v14 != v23 )
  {
LABEL_17:
    if ( *v4 > 0x15u )
    {
      v24 = v11;
      v17 = v15;
      v9 = sub_AD93D0(v15, *((_QWORD *)v4 + 1), 0, 0);
      v11 = v24;
      if ( v9 )
      {
        v10 = v17;
        v8 = v4;
        goto LABEL_6;
      }
    }
    goto LABEL_13;
  }
  v18 = v23;
  v25 = v14;
  result = sub_F0CA40(a1, a2, (const __m128i *)(a1 + 96), *(__int64 **)(a1 + 32), v14, v26, v27, v28, v29[0]);
  if ( !result )
  {
    v11 = v25;
    v15 = v18;
    goto LABEL_17;
  }
  return result;
}
