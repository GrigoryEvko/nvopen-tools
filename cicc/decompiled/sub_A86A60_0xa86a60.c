// Function: sub_A86A60
// Address: 0xa86a60
//
__int64 __fastcall sub_A86A60(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // r15
  unsigned __int64 v3; // rdx
  const __m128i *v4; // rbx
  _BYTE *v5; // rax
  _QWORD *v6; // r15
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rcx
  __int64 v10; // rax
  size_t v11; // rcx
  void *v12; // rdi
  unsigned __int64 v13; // rdx
  size_t v14; // rdx
  size_t v15; // r12
  __int64 v16; // rdx
  __int64 v17; // rcx
  _BYTE *v18; // rsi
  __int64 v19; // [rsp+8h] [rbp-188h]
  __int64 v20; // [rsp+20h] [rbp-170h]
  __int64 i; // [rsp+28h] [rbp-168h]
  const __m128i *v22; // [rsp+30h] [rbp-160h]
  _QWORD v23[2]; // [rsp+40h] [rbp-150h] BYREF
  __m128i v24; // [rsp+50h] [rbp-140h] BYREF
  void *src; // [rsp+60h] [rbp-130h] BYREF
  unsigned __int64 v26; // [rsp+68h] [rbp-128h]
  _QWORD v27[2]; // [rsp+70h] [rbp-120h] BYREF
  _QWORD v28[3]; // [rsp+80h] [rbp-110h] BYREF
  _BYTE v29[40]; // [rsp+98h] [rbp-F8h] BYREF
  _QWORD v30[3]; // [rsp+C0h] [rbp-D0h] BYREF
  unsigned __int64 v31; // [rsp+D8h] [rbp-B8h]
  _BYTE *v32; // [rsp+E0h] [rbp-B0h]
  __int64 v33; // [rsp+E8h] [rbp-A8h]
  __int64 *v34; // [rsp+F0h] [rbp-A0h]
  const __m128i *v35; // [rsp+100h] [rbp-90h] BYREF
  __int64 v36; // [rsp+108h] [rbp-88h]
  _BYTE v37[128]; // [rsp+110h] [rbp-80h] BYREF

  result = a1 + 8;
  v2 = *(_QWORD *)(a1 + 16);
  for ( i = a1 + 8; i != v2; v2 = *(_QWORD *)(v2 + 8) )
  {
    while ( 1 )
    {
      if ( !v2 )
        BUG();
      if ( (*(_BYTE *)(v2 - 21) & 4) != 0 )
      {
        v20 = v2 - 56;
        result = sub_B31D10(v2 - 56);
        if ( v3 > 0x15
          && !(*(_QWORD *)result ^ 0x202C415441445F5FLL | *(_QWORD *)(result + 8) ^ 0x635F636A626F5F5FLL)
          && *(_DWORD *)(result + 16) == 1768715361
          && *(_WORD *)(result + 20) == 29811 )
        {
          v23[0] = result;
          v23[1] = v3;
          v35 = (const __m128i *)v37;
          v36 = 0x500000000LL;
          sub_C93960(v23, &v35, 44, 0xFFFFFFFFLL, 1);
          v34 = v28;
          v28[0] = v29;
          v30[0] = &unk_49DD288;
          v33 = 0x100000000LL;
          v28[1] = 0;
          v28[2] = 32;
          v30[1] = 2;
          v30[2] = 0;
          v31 = 0;
          v32 = 0;
          sub_CB5980(v30, 0, 0, 0);
          v4 = v35;
          v22 = &v35[(unsigned int)v36];
          if ( v35 != v22 )
          {
            v19 = v2;
            do
            {
              while ( 1 )
              {
                v5 = v32;
                v24 = _mm_loadu_si128(v4);
                if ( (unsigned __int64)v32 >= v31 )
                {
                  v6 = (_QWORD *)sub_CB5D20(v30, 44);
                }
                else
                {
                  v6 = v30;
                  ++v32;
                  *v5 = 44;
                }
                v7 = 0;
                v8 = sub_C935B0(&v24, &unk_3F15413, 6, 0);
                v9 = v24.m128i_u64[1];
                if ( v8 < v24.m128i_i64[1] )
                {
                  v7 = v24.m128i_i64[1] - v8;
                  v9 = v8;
                }
                v26 = v7;
                src = (void *)(v24.m128i_i64[0] + v9);
                v10 = sub_C93740(&src, &unk_3F15413, 6, -1);
                v11 = v26;
                v12 = (void *)v6[4];
                v13 = v10 + 1;
                if ( v10 + 1 > v26 )
                  v13 = v26;
                v14 = v26 - v7 + v13;
                if ( v14 <= v26 )
                  v11 = v14;
                v15 = v11;
                if ( v6[3] - (_QWORD)v12 >= v11 )
                  break;
                ++v4;
                sub_CB6200(v6, src, v11);
                if ( v22 == v4 )
                  goto LABEL_25;
              }
              if ( v11 )
              {
                memcpy(v12, src, v11);
                v6[4] += v15;
              }
              ++v4;
            }
            while ( v22 != v4 );
LABEL_25:
            v2 = v19;
          }
          v16 = *v34;
          v17 = v34[1];
          v18 = (_BYTE *)*v34;
          if ( v17 )
          {
            v18 = (_BYTE *)(v16 + 1);
            v16 += v17;
          }
          src = v27;
          sub_A7BD10((__int64 *)&src, v18, v16);
          v30[0] = &unk_49DD388;
          sub_CB5840(v30);
          if ( (_BYTE *)v28[0] != v29 )
            _libc_free(v28[0], v18);
          if ( v35 != (const __m128i *)v37 )
            _libc_free(v35, v18);
          result = sub_B31A00(v20, src, v26);
          if ( src != v27 )
            break;
        }
      }
      v2 = *(_QWORD *)(v2 + 8);
      if ( i == v2 )
        return result;
    }
    result = j_j___libc_free_0(src, v27[0] + 1LL);
  }
  return result;
}
