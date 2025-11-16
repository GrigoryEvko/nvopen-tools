// Function: sub_C6C710
// Address: 0xc6c710
//
unsigned __int64 __fastcall sub_C6C710(__int64 a1, unsigned __int16 *a2, __int64 a3)
{
  __int64 v3; // r13
  unsigned __int64 result; // rax
  __int64 v6; // r13
  __int64 i; // rbx
  __int64 v8; // rsi
  __int64 *v9; // rsi
  __int64 *v10; // r13
  __int64 *v11; // r14
  __int64 v12; // rbx
  __int64 v13; // rdi
  _DWORD *v14; // rdx
  char v15; // al
  const char *v16; // r13
  __int64 v17; // r12
  size_t v18; // rax
  _QWORD *v19; // rcx
  unsigned __int64 v20; // rdx
  char *v21; // rsi
  __int16 v22; // ax
  __int64 v23; // rdi
  __int64 v24; // rax
  unsigned __int64 v25; // rdi
  char *v26; // rcx
  const char *v27; // r13
  unsigned int v28; // ecx
  __int64 v29; // rsi
  __m128i v30; // [rsp-48h] [rbp-48h] BYREF
  __int64 v31; // [rsp-38h] [rbp-38h]
  int v32; // [rsp-30h] [rbp-30h]

  result = *a2;
  if ( (unsigned __int16)result > 8u )
    BUG();
  if ( *(_DWORD *)(a3 + 4 * result) <= 5u )
  {
    switch ( (__int16)result )
    {
      case 0:
        sub_C6AAB0(a1);
        v13 = *(_QWORD *)(a1 + 160);
        v14 = *(_DWORD **)(v13 + 32);
        result = *(_QWORD *)(v13 + 24) - (_QWORD)v14;
        if ( result <= 3 )
        {
          v20 = 4;
          v21 = "null";
          goto LABEL_24;
        }
        *v14 = 1819047278;
        *(_QWORD *)(v13 + 32) += 4LL;
        break;
      case 1:
        sub_C6AAB0(a1);
        v15 = 0;
        if ( *a2 == 1 )
          v15 = *((_BYTE *)a2 + 8);
        v16 = "true";
        v17 = *(_QWORD *)(a1 + 160);
        if ( !v15 )
          v16 = "false";
        v18 = strlen(v16);
        v19 = *(_QWORD **)(v17 + 32);
        v20 = v18;
        result = *(_QWORD *)(v17 + 24) - (_QWORD)v19;
        if ( v20 <= result )
        {
          if ( (unsigned int)v20 >= 8 )
          {
            v25 = (unsigned __int64)(v19 + 1) & 0xFFFFFFFFFFFFFFF8LL;
            *v19 = *(_QWORD *)v16;
            *(_QWORD *)((char *)v19 + (unsigned int)v20 - 8) = *(_QWORD *)&v16[(unsigned int)v20 - 8];
            v26 = (char *)v19 - v25;
            v27 = (const char *)(v16 - v26);
            result = ((_DWORD)v20 + (_DWORD)v26) & 0xFFFFFFF8;
            if ( (unsigned int)result >= 8 )
            {
              result = ((_DWORD)v20 + (_DWORD)v26) & 0xFFFFFFF8;
              v28 = 0;
              do
              {
                v29 = v28;
                v28 += 8;
                *(_QWORD *)(v25 + v29) = *(_QWORD *)&v27[v29];
              }
              while ( v28 < (unsigned int)result );
            }
          }
          else if ( (v20 & 4) != 0 )
          {
            *(_DWORD *)v19 = *(_DWORD *)v16;
            result = (unsigned int)v20;
            *(_DWORD *)((char *)v19 + (unsigned int)v20 - 4) = *(_DWORD *)&v16[(unsigned int)v20 - 4];
          }
          else if ( (_DWORD)v20 )
          {
            result = *(unsigned __int8 *)v16;
            *(_BYTE *)v19 = result;
            if ( (v20 & 2) != 0 )
            {
              result = (unsigned int)v20;
              *(_WORD *)((char *)v19 + (unsigned int)v20 - 2) = *(_WORD *)&v16[(unsigned int)v20 - 2];
            }
          }
          *(_QWORD *)(v17 + 32) += v20;
        }
        else
        {
          v21 = (char *)v16;
          v13 = v17;
LABEL_24:
          result = sub_CB6200(v13, v21, v20);
        }
        break;
      case 2:
      case 3:
      case 4:
        sub_C6AAB0(a1);
        v22 = *a2;
        if ( *a2 == 3 )
        {
          result = sub_CB59F0(*(_QWORD *)(a1 + 160), *((_QWORD *)a2 + 1));
        }
        else if ( v22 == 4 )
        {
          result = sub_CB59D0(*(_QWORD *)(a1 + 160), *((_QWORD *)a2 + 1));
        }
        else
        {
          v23 = *(_QWORD *)(a1 + 160);
          if ( v22 == 2 )
            v3 = *((_QWORD *)a2 + 1);
          v31 = v3;
          v30.m128i_i64[1] = (__int64)"%.*g";
          v32 = 17;
          v30.m128i_i64[0] = (__int64)&unk_49DC970;
          result = sub_CB6620(v23, &v30);
        }
        break;
      case 5:
      case 6:
        sub_C6AAB0(a1);
        if ( *a2 == 6 )
        {
          v24 = *((_QWORD *)a2 + 2);
          v30.m128i_i64[0] = *((_QWORD *)a2 + 1);
          v30.m128i_i64[1] = v24;
        }
        else if ( *a2 == 5 )
        {
          v30 = _mm_loadu_si128((const __m128i *)(a2 + 4));
        }
        result = (unsigned __int64)sub_C69320(
                                     *(_QWORD *)(a1 + 160),
                                     (unsigned __int8 *)v30.m128i_i64[0],
                                     v30.m128i_i64[1]);
        break;
      case 7:
        sub_C6ACB0(a1);
        v9 = (__int64 *)(a2 + 4);
        if ( *a2 != 7 )
          v9 = 0;
        sub_C6C520((char **)&v30, v9);
        v11 = (__int64 *)v30.m128i_i64[1];
        v10 = (__int64 *)v30.m128i_i64[0];
        if ( v30.m128i_i64[0] != v30.m128i_i64[1] )
        {
          do
          {
            v12 = *v10++;
            sub_C6B410(a1, *(unsigned __int8 **)(v12 + 8), *(_QWORD *)(v12 + 16));
            sub_C6C710(a1, v12 + 24);
            sub_C6AE10(a1);
          }
          while ( v11 != v10 );
          v11 = (__int64 *)v30.m128i_i64[0];
        }
        if ( v11 )
          j_j___libc_free_0(v11, v31 - (_QWORD)v11);
        result = (unsigned __int64)sub_C6AD90(a1);
        break;
      case 8:
        sub_C6AB50(a1);
        if ( *a2 != 8 )
          BUG();
        v6 = *((_QWORD *)a2 + 2);
        for ( i = *((_QWORD *)a2 + 1); v6 != i; i += 40 )
        {
          v8 = i;
          sub_C6C710(a1, v8);
        }
        result = (unsigned __int64)sub_C6AC30(a1);
        break;
    }
  }
  return result;
}
