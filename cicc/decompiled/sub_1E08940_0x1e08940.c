// Function: sub_1E08940
// Address: 0x1e08940
//
unsigned __int64 __fastcall sub_1E08940(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 result; // rax
  __int64 (*v6)(); // rax
  int v7; // r12d
  void *v8; // rdx
  unsigned int v9; // edx
  __int64 v10; // rbx
  _QWORD *v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // r12
  __m128i si128; // xmm0
  _BYTE *v15; // rax
  unsigned __int64 v16; // r12
  __int64 v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rax
  _WORD *v20; // rdx
  _BYTE *v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rdi
  __int64 v28; // rdi
  _BYTE *v29; // rax
  _BYTE *v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // [rsp-48h] [rbp-48h]
  __int64 v34; // [rsp-40h] [rbp-40h]

  result = *(_QWORD *)(a1 + 16);
  if ( *(_QWORD *)(a1 + 8) != result )
  {
    v6 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 48LL);
    if ( v6 == sub_1D90020 || (v32 = v6()) == 0 )
      v7 = 0;
    else
      v7 = *(_DWORD *)(v32 + 20);
    v8 = *(void **)(a3 + 24);
    if ( *(_QWORD *)(a3 + 16) - (_QWORD)v8 <= 0xEu )
    {
      sub_16E7EE0(a3, "Frame Objects:\n", 0xFu);
    }
    else
    {
      qmemcpy(v8, "Frame Objects:\n", 15);
      *(_QWORD *)(a3 + 24) += 15LL;
    }
    result = *(_QWORD *)(a1 + 8);
    v9 = -858993459 * ((__int64)(*(_QWORD *)(a1 + 16) - result) >> 3);
    if ( v9 )
    {
      v10 = 0;
      v34 = v9;
      v33 = v7;
      while ( 1 )
      {
        v16 = result + 40 * v10;
        v17 = *(_QWORD *)(a3 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v17) <= 4 )
        {
          v18 = sub_16E7EE0(a3, "  fi#", 5u);
        }
        else
        {
          *(_DWORD *)v17 = 1768300576;
          v18 = a3;
          *(_BYTE *)(v17 + 4) = 35;
          *(_QWORD *)(a3 + 24) += 5LL;
        }
        v19 = sub_16E7AB0(v18, (int)v10 - *(_DWORD *)(a1 + 32));
        v20 = *(_WORD **)(v19 + 24);
        if ( *(_QWORD *)(v19 + 16) - (_QWORD)v20 <= 1u )
        {
          sub_16E7EE0(v19, ": ", 2u);
        }
        else
        {
          *v20 = 8250;
          *(_QWORD *)(v19 + 24) += 2LL;
        }
        if ( *(_BYTE *)(v16 + 23) )
        {
          v26 = *(_QWORD *)(a3 + 24);
          if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v26) <= 2 )
          {
            v27 = sub_16E7EE0(a3, "id=", 3u);
          }
          else
          {
            *(_BYTE *)(v26 + 2) = 61;
            v27 = a3;
            *(_WORD *)v26 = 25705;
            *(_QWORD *)(a3 + 24) += 3LL;
          }
          v28 = sub_16E7A90(v27, *(unsigned __int8 *)(v16 + 23));
          v29 = *(_BYTE **)(v28 + 24);
          if ( (unsigned __int64)v29 >= *(_QWORD *)(v28 + 16) )
          {
            sub_16E7DE0(v28, 32);
          }
          else
          {
            *(_QWORD *)(v28 + 24) = v29 + 1;
            *v29 = 32;
          }
        }
        v21 = *(_BYTE **)(a3 + 24);
        v22 = *(_QWORD *)(v16 + 8);
        result = *(_QWORD *)(a3 + 16) - (_QWORD)v21;
        if ( v22 == -1 )
        {
          if ( result <= 4 )
          {
            result = sub_16E7EE0(a3, "dead\n", 5u);
          }
          else
          {
            *(_DWORD *)v21 = 1684104548;
            v21[4] = 10;
            *(_QWORD *)(a3 + 24) += 5LL;
          }
LABEL_23:
          if ( ++v10 == v34 )
            return result;
          goto LABEL_24;
        }
        if ( v22 )
        {
          if ( result <= 4 )
          {
            v23 = sub_16E7EE0(a3, "size=", 5u);
          }
          else
          {
            *(_DWORD *)v21 = 1702521203;
            v23 = a3;
            v21[4] = 61;
            *(_QWORD *)(a3 + 24) += 5LL;
          }
          sub_16E7A90(v23, *(_QWORD *)(v16 + 8));
          v11 = *(_QWORD **)(a3 + 24);
          if ( *(_QWORD *)(a3 + 16) - (_QWORD)v11 > 7u )
            goto LABEL_11;
        }
        else
        {
          if ( result <= 0xD )
          {
            sub_16E7EE0(a3, "variable sized", 0xEu);
            v11 = *(_QWORD **)(a3 + 24);
          }
          else
          {
            qmemcpy(v21, "variable sized", 14);
            v11 = (_QWORD *)(*(_QWORD *)(a3 + 24) + 14LL);
            *(_QWORD *)(a3 + 24) = v11;
          }
          if ( *(_QWORD *)(a3 + 16) - (_QWORD)v11 > 7u )
          {
LABEL_11:
            *v11 = 0x3D6E67696C61202CLL;
            *(_QWORD *)(a3 + 24) += 8LL;
            sub_16E7A90(a3, *(unsigned int *)(v16 + 16));
            if ( *(_DWORD *)(a1 + 32) <= (unsigned int)v10 )
              goto LABEL_12;
            goto LABEL_36;
          }
        }
        v24 = sub_16E7EE0(a3, ", align=", 8u);
        sub_16E7A90(v24, *(unsigned int *)(v16 + 16));
        if ( *(_DWORD *)(a1 + 32) <= (unsigned int)v10 )
        {
LABEL_12:
          v12 = *(_QWORD *)v16;
          goto LABEL_13;
        }
LABEL_36:
        v25 = *(_QWORD *)(a3 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v25) <= 6 )
        {
          sub_16E7EE0(a3, ", fixed", 7u);
          result = *(_QWORD *)(a3 + 24);
          v12 = *(_QWORD *)v16;
          if ( *(_DWORD *)(a1 + 32) > (unsigned int)v10 )
            goto LABEL_14;
        }
        else
        {
          *(_DWORD *)v25 = 1768300588;
          *(_WORD *)(v25 + 4) = 25976;
          *(_BYTE *)(v25 + 6) = 100;
          result = *(_QWORD *)(a3 + 24) + 7LL;
          *(_QWORD *)(a3 + 24) = result;
          v12 = *(_QWORD *)v16;
          if ( *(_DWORD *)(a1 + 32) > (unsigned int)v10 )
            goto LABEL_14;
        }
LABEL_13:
        result = *(_QWORD *)(a3 + 24);
        if ( v12 == -1 )
          goto LABEL_21;
LABEL_14:
        v13 = v12 - v33;
        if ( *(_QWORD *)(a3 + 16) - result <= 0x10 )
        {
          sub_16E7EE0(a3, ", at location [SP", 0x11u);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_42EAC70);
          *(_BYTE *)(result + 16) = 80;
          *(__m128i *)result = si128;
          *(_QWORD *)(a3 + 24) += 17LL;
        }
        if ( v13 > 0 )
        {
          v30 = *(_BYTE **)(a3 + 24);
          if ( *(_BYTE **)(a3 + 16) == v30 )
          {
            v31 = sub_16E7EE0(a3, "+", 1u);
            sub_16E7AB0(v31, v13);
          }
          else
          {
            *v30 = 43;
            ++*(_QWORD *)(a3 + 24);
            sub_16E7AB0(a3, v13);
          }
          v15 = *(_BYTE **)(a3 + 24);
          if ( *(_BYTE **)(a3 + 16) != v15 )
          {
LABEL_20:
            *v15 = 93;
            result = *(_QWORD *)(a3 + 24) + 1LL;
            *(_QWORD *)(a3 + 24) = result;
            goto LABEL_21;
          }
        }
        else
        {
          if ( v13 )
            sub_16E7AB0(a3, v13);
          v15 = *(_BYTE **)(a3 + 24);
          if ( *(_BYTE **)(a3 + 16) != v15 )
            goto LABEL_20;
        }
        sub_16E7EE0(a3, "]", 1u);
        result = *(_QWORD *)(a3 + 24);
LABEL_21:
        if ( result != *(_QWORD *)(a3 + 16) )
        {
          *(_BYTE *)result = 10;
          ++*(_QWORD *)(a3 + 24);
          goto LABEL_23;
        }
        ++v10;
        result = sub_16E7EE0(a3, "\n", 1u);
        if ( v10 == v34 )
          return result;
LABEL_24:
        result = *(_QWORD *)(a1 + 8);
      }
    }
  }
  return result;
}
