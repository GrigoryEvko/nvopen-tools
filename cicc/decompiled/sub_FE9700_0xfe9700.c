// Function: sub_FE9700
// Address: 0xfe9700
//
unsigned int *__fastcall sub_FE9700(_QWORD *a1)
{
  unsigned __int64 v1; // r13
  __int64 v3; // rdx
  __int64 v4; // r15
  __int64 v5; // rax
  __int16 v6; // dx
  __int16 v7; // di
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 *v10; // r15
  unsigned int *result; // rax
  __int64 v12; // rdx
  unsigned int *v13; // r8
  __int64 v14; // rax
  unsigned int *v15; // r13
  __int64 v16; // rbx
  __int16 v17; // ax
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 **v20; // rbx
  __int64 v21; // rax
  __int64 *v22; // rdi
  __int64 *v23; // rax
  __int64 *i; // rbx
  bool v25; // al
  __int64 v26; // [rsp+10h] [rbp-70h]
  unsigned int *v27; // [rsp+18h] [rbp-68h]
  __m128i v28; // [rsp+20h] [rbp-60h]
  __int64 v29; // [rsp+40h] [rbp-40h] BYREF
  __int64 v30; // [rsp+48h] [rbp-38h]

  v1 = 0;
  v3 = a1[8];
  if ( v3 != a1[9] )
  {
    do
    {
      v4 = 3 * v1++;
      v5 = sub_FE8600((_QWORD *)(v3 + 8 * v4 + 16));
      v7 = v6;
      v8 = v5;
      v9 = a1[1];
      *(_QWORD *)(v9 + 8 * v4) = v8;
      *(_WORD *)(v9 + 8 * v4 + 8) = v7;
      v3 = a1[8];
    }
    while ( 0xAAAAAAAAAAAAAAABLL * ((a1[9] - v3) >> 3) > v1 );
  }
  v10 = (__int64 *)a1[11];
  result = (unsigned int *)(a1 + 11);
  if ( a1 + 11 != v10 )
  {
    while ( 1 )
    {
      v29 = sub_FE8600(v10 + 21);
      v30 = v12;
      sub_FE9650((__int64)(v10 + 22), (__int64)&v29);
      v13 = (unsigned int *)v10[14];
      v14 = *((unsigned int *)v10 + 30);
      *((_BYTE *)v10 + 24) = 0;
      result = &v13[v14];
      v15 = v13;
      v27 = result;
      if ( v13 != result )
        break;
LABEL_18:
      v10 = (__int64 *)*v10;
      if ( a1 + 11 == v10 )
        return result;
    }
    while ( 1 )
    {
      v18 = 24LL * *v15;
      v19 = v18 + a1[8];
      v20 = *(__int64 ***)(v19 + 8);
      if ( v20 )
      {
        v21 = *((unsigned int *)v20 + 3);
        v22 = v20[12];
        if ( (unsigned int)v21 > 1 )
        {
          v26 = 24LL * *v15;
          v25 = sub_FDC990(v22, (_DWORD *)v22 + v21, (_DWORD *)v19);
          v18 = v26;
          if ( !v25 )
            goto LABEL_6;
        }
        else if ( *(_DWORD *)v19 != *(_DWORD *)v22 )
        {
          goto LABEL_6;
        }
        if ( *((_BYTE *)v20 + 8) )
        {
          v23 = (__int64 *)v20;
          for ( i = *v20; i; i = (__int64 *)*i )
          {
            if ( !*((_BYTE *)i + 8) )
              break;
            v23 = i;
          }
          v16 = (__int64)(v23 + 20);
          goto LABEL_7;
        }
      }
LABEL_6:
      v16 = a1[1] + v18;
LABEL_7:
      v17 = *((_WORD *)v10 + 92);
      ++v15;
      v29 = v10[22];
      LOWORD(v30) = v17;
      v28 = _mm_loadu_si128((const __m128i *)sub_FE9650((__int64)&v29, v16));
      *(_QWORD *)v16 = v28.m128i_i64[0];
      result = (unsigned int *)v28.m128i_u16[4];
      *(_WORD *)(v16 + 8) = v28.m128i_i16[4];
      if ( v27 == v15 )
        goto LABEL_18;
    }
  }
  return result;
}
