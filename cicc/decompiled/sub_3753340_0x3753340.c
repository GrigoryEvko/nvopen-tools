// Function: sub_3753340
// Address: 0x3753340
//
__int64 __fastcall sub_3753340(__int64 *a1, __int64 *a2, __int64 a3, unsigned int *a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  unsigned int *v7; // r13
  unsigned int *v10; // rbx
  unsigned __int64 v12; // rdx
  unsigned int v13; // esi
  __int64 v14; // r9
  int v15; // ecx
  int v16; // r11d
  unsigned int i; // eax
  __int64 v18; // r8
  unsigned int v19; // eax
  __int32 v20; // eax
  __int64 v21; // rdi
  __int64 v22; // rsi
  unsigned int v23; // eax
  __int64 v24; // rdi
  __int64 v25; // rsi
  int v26; // ecx
  __int64 v27; // rdi
  __int64 v28; // rsi
  __m128i v30; // [rsp+10h] [rbp-60h] BYREF
  __int64 v31; // [rsp+20h] [rbp-50h]
  __int64 v32; // [rsp+28h] [rbp-48h]
  __int64 v33; // [rsp+30h] [rbp-40h]

  result = 3 * a5;
  v7 = &a4[6 * a5];
  if ( v7 != a4 )
  {
    v10 = a4;
    while ( 1 )
    {
      result = *v10;
      if ( (_DWORD)result == 2 )
      {
        v23 = v10[2];
        v24 = a2[1];
        v30.m128i_i64[0] = 5;
        v25 = *a2;
        v31 = 0;
        LODWORD(v32) = v23;
        result = sub_2E8EAD0(v24, v25, &v30);
        goto LABEL_4;
      }
      if ( (unsigned int)result > 2 )
      {
        if ( (_DWORD)result != 3 )
          goto LABEL_4;
        v20 = v10[2];
        v21 = a2[1];
        v10 += 6;
        v30.m128i_i64[0] = 0;
        v22 = *a2;
        v31 = 0;
        v30.m128i_i32[2] = v20;
        v32 = 0;
        v33 = 0;
        result = sub_2E8EAD0(v21, v22, &v30);
        if ( v7 == v10 )
          return result;
      }
      else
      {
        if ( (_DWORD)result )
        {
          sub_3753270(&v30, (__int64)v10);
          result = sub_2E8EAD0(a2[1], *a2, &v30);
        }
        else
        {
          v12 = *((_QWORD *)v10 + 1);
          v13 = v10[4];
          if ( (*(_BYTE *)(a6 + 8) & 1) != 0 )
          {
            v14 = a6 + 16;
            v15 = 15;
            goto LABEL_10;
          }
          v26 = *(_DWORD *)(a6 + 24);
          v14 = *(_QWORD *)(a6 + 16);
          if ( v26 )
          {
            v15 = v26 - 1;
LABEL_10:
            v16 = 1;
            for ( i = v15 & (v13 + ((v12 >> 9) ^ (v12 >> 4))); ; i = v15 & v19 )
            {
              v18 = v14 + 24LL * i;
              if ( v12 == *(_QWORD *)v18 && v13 == *(_DWORD *)(v18 + 8) )
                break;
              if ( !*(_QWORD *)v18 && *(_DWORD *)(v18 + 8) == -1 )
                goto LABEL_21;
              v19 = v16 + i;
              ++v16;
            }
            result = sub_3752760(a1, a2, v12, v10[4], *(_DWORD *)(a2[1] + 40) & 0xFFFFFF, a3, a6, 1, 0, 0);
            goto LABEL_4;
          }
LABEL_21:
          v27 = a2[1];
          v28 = *a2;
          v30 = 0u;
          v31 = 0;
          v32 = 0;
          v33 = 0;
          result = sub_2E8EAD0(v27, v28, &v30);
        }
LABEL_4:
        v10 += 6;
        if ( v7 == v10 )
          return result;
      }
    }
  }
  return result;
}
