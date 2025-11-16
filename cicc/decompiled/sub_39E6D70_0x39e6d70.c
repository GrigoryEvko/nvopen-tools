// Function: sub_39E6D70
// Address: 0x39e6d70
//
_BYTE *__fastcall sub_39E6D70(__int64 a1, int a2)
{
  _BYTE *result; // rax
  __int64 v4; // rdi
  size_t v5; // r13
  __int64 v6; // rdi
  __int64 v7; // r14
  char *v8; // rsi
  void *v9; // rdi
  __m128i *v10; // rdx
  __m128i si128; // xmm0

  result = *(_BYTE **)(a1 + 280);
  if ( result[174] )
  {
    v4 = *(_QWORD *)(a1 + 272);
    switch ( a2 )
    {
      case 0:
        sub_1263B40(v4, "\t.data_region");
        v5 = *(unsigned int *)(a1 + 312);
        if ( *(_DWORD *)(a1 + 312) )
          goto LABEL_9;
        goto LABEL_5;
      case 1:
        sub_1263B40(v4, "\t.data_region jt8");
        v5 = *(unsigned int *)(a1 + 312);
        if ( *(_DWORD *)(a1 + 312) )
          goto LABEL_9;
        goto LABEL_5;
      case 2:
        sub_1263B40(v4, "\t.data_region jt16");
        v5 = *(unsigned int *)(a1 + 312);
        if ( *(_DWORD *)(a1 + 312) )
          goto LABEL_9;
        goto LABEL_5;
      case 3:
        sub_1263B40(v4, "\t.data_region jt32");
        goto LABEL_4;
      case 4:
        v10 = *(__m128i **)(v4 + 24);
        if ( *(_QWORD *)(v4 + 16) - (_QWORD)v10 <= 0x10u )
        {
          sub_16E7EE0(v4, "\t.end_data_region", 0x11u);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_4534D30);
          v10[1].m128i_i8[0] = 110;
          *v10 = si128;
          *(_QWORD *)(v4 + 24) += 17LL;
        }
        goto LABEL_4;
      default:
LABEL_4:
        v5 = *(unsigned int *)(a1 + 312);
        if ( *(_DWORD *)(a1 + 312) )
        {
LABEL_9:
          v7 = *(_QWORD *)(a1 + 272);
          v8 = *(char **)(a1 + 304);
          v9 = *(void **)(v7 + 24);
          if ( v5 > *(_QWORD *)(v7 + 16) - (_QWORD)v9 )
          {
            sub_16E7EE0(*(_QWORD *)(a1 + 272), v8, v5);
          }
          else
          {
            memcpy(v9, v8, v5);
            *(_QWORD *)(v7 + 24) += v5;
          }
        }
LABEL_5:
        *(_DWORD *)(a1 + 312) = 0;
        if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
        {
          result = sub_39E0440(a1);
        }
        else
        {
          v6 = *(_QWORD *)(a1 + 272);
          result = *(_BYTE **)(v6 + 24);
          if ( (unsigned __int64)result >= *(_QWORD *)(v6 + 16) )
          {
            result = (_BYTE *)sub_16E7DE0(v6, 10);
          }
          else
          {
            *(_QWORD *)(v6 + 24) = result + 1;
            *result = 10;
          }
        }
        break;
    }
  }
  return result;
}
