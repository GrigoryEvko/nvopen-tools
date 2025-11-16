// Function: sub_2E77130
// Address: 0x2e77130
//
unsigned __int64 __fastcall sub_2E77130(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 result; // rax
  __int64 (*v6)(); // rax
  void *v7; // rdx
  unsigned int v8; // edx
  __int64 v9; // rbx
  _QWORD *v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rdx
  signed __int64 v13; // r12
  __m128i si128; // xmm0
  _BYTE *v15; // rax
  unsigned __int64 v16; // r12
  __int64 v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rax
  _WORD *v20; // rdx
  _BYTE *v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rdi
  __int64 v26; // rdi
  _BYTE *v27; // rax
  __int64 v28; // rdx
  _BYTE *v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  int v32; // [rsp-44h] [rbp-44h]
  __int64 v33; // [rsp-40h] [rbp-40h]

  result = *(_QWORD *)(a1 + 16);
  if ( *(_QWORD *)(a1 + 8) != result )
  {
    v6 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 136LL);
    if ( v6 == sub_2DD19D0 || (v31 = v6()) == 0 )
      v32 = 0;
    else
      v32 = *(_DWORD *)(v31 + 16);
    v7 = *(void **)(a3 + 32);
    if ( *(_QWORD *)(a3 + 24) - (_QWORD)v7 <= 0xEu )
    {
      sub_CB6200(a3, "Frame Objects:\n", 0xFu);
    }
    else
    {
      qmemcpy(v7, "Frame Objects:\n", 15);
      *(_QWORD *)(a3 + 32) += 15LL;
    }
    result = *(_QWORD *)(a1 + 8);
    v8 = -858993459 * ((__int64)(*(_QWORD *)(a1 + 16) - result) >> 3);
    if ( v8 )
    {
      v33 = v8;
      v9 = 0;
      while ( 1 )
      {
        v16 = result + 40 * v9;
        v17 = *(_QWORD *)(a3 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a3 + 24) - v17) <= 4 )
        {
          v18 = sub_CB6200(a3, "  fi#", 5u);
        }
        else
        {
          *(_DWORD *)v17 = 1768300576;
          v18 = a3;
          *(_BYTE *)(v17 + 4) = 35;
          *(_QWORD *)(a3 + 32) += 5LL;
        }
        v19 = sub_CB59F0(v18, (int)v9 - *(_DWORD *)(a1 + 32));
        v20 = *(_WORD **)(v19 + 32);
        if ( *(_QWORD *)(v19 + 24) - (_QWORD)v20 <= 1u )
        {
          sub_CB6200(v19, (unsigned __int8 *)": ", 2u);
        }
        else
        {
          *v20 = 8250;
          *(_QWORD *)(v19 + 32) += 2LL;
        }
        if ( *(_BYTE *)(v16 + 20) )
        {
          v24 = *(_QWORD *)(a3 + 32);
          if ( (unsigned __int64)(*(_QWORD *)(a3 + 24) - v24) <= 2 )
          {
            v25 = sub_CB6200(a3, "id=", 3u);
          }
          else
          {
            *(_BYTE *)(v24 + 2) = 61;
            v25 = a3;
            *(_WORD *)v24 = 25705;
            *(_QWORD *)(a3 + 32) += 3LL;
          }
          v26 = sub_CB59D0(v25, *(unsigned __int8 *)(v16 + 20));
          v27 = *(_BYTE **)(v26 + 32);
          if ( (unsigned __int64)v27 >= *(_QWORD *)(v26 + 24) )
          {
            sub_CB5D20(v26, 32);
          }
          else
          {
            *(_QWORD *)(v26 + 32) = v27 + 1;
            *v27 = 32;
          }
        }
        v21 = *(_BYTE **)(a3 + 32);
        v22 = *(_QWORD *)(v16 + 8);
        result = *(_QWORD *)(a3 + 24) - (_QWORD)v21;
        if ( v22 == -1 )
        {
          if ( result <= 4 )
          {
            result = sub_CB6200(a3, "dead\n", 5u);
          }
          else
          {
            *(_DWORD *)v21 = 1684104548;
            v21[4] = 10;
            *(_QWORD *)(a3 + 32) += 5LL;
          }
LABEL_24:
          if ( ++v9 == v33 )
            return result;
          goto LABEL_25;
        }
        if ( v22 )
        {
          if ( result <= 4 )
          {
            v23 = sub_CB6200(a3, (unsigned __int8 *)"size=", 5u);
          }
          else
          {
            *(_DWORD *)v21 = 1702521203;
            v23 = a3;
            v21[4] = 61;
            *(_QWORD *)(a3 + 32) += 5LL;
          }
          sub_CB59D0(v23, *(_QWORD *)(v16 + 8));
          v10 = *(_QWORD **)(a3 + 32);
          if ( *(_QWORD *)(a3 + 24) - (_QWORD)v10 > 7u )
            goto LABEL_11;
        }
        else
        {
          if ( result <= 0xD )
          {
            sub_CB6200(a3, "variable sized", 0xEu);
            v10 = *(_QWORD **)(a3 + 32);
          }
          else
          {
            qmemcpy(v21, "variable sized", 14);
            v10 = (_QWORD *)(*(_QWORD *)(a3 + 32) + 14LL);
            *(_QWORD *)(a3 + 32) = v10;
          }
          if ( *(_QWORD *)(a3 + 24) - (_QWORD)v10 > 7u )
          {
LABEL_11:
            v11 = a3;
            *v10 = 0x3D6E67696C61202CLL;
            *(_QWORD *)(a3 + 32) += 8LL;
            goto LABEL_12;
          }
        }
        v11 = sub_CB6200(a3, ", align=", 8u);
LABEL_12:
        sub_CB59D0(v11, 1LL << *(_BYTE *)(v16 + 16));
        if ( *(_DWORD *)(a1 + 32) > (unsigned int)v9 )
        {
          v28 = *(_QWORD *)(a3 + 32);
          if ( (unsigned __int64)(*(_QWORD *)(a3 + 24) - v28) <= 6 )
          {
            sub_CB6200(a3, ", fixed", 7u);
            result = *(_QWORD *)(a3 + 32);
            v12 = *(_QWORD *)v16;
            if ( *(_DWORD *)(a1 + 32) > (unsigned int)v9 )
              goto LABEL_15;
          }
          else
          {
            *(_DWORD *)v28 = 1768300588;
            *(_WORD *)(v28 + 4) = 25976;
            *(_BYTE *)(v28 + 6) = 100;
            result = *(_QWORD *)(a3 + 32) + 7LL;
            *(_QWORD *)(a3 + 32) = result;
            v12 = *(_QWORD *)v16;
            if ( *(_DWORD *)(a1 + 32) > (unsigned int)v9 )
              goto LABEL_15;
          }
        }
        else
        {
          v12 = *(_QWORD *)v16;
        }
        result = *(_QWORD *)(a3 + 32);
        if ( v12 == -1 )
          goto LABEL_22;
LABEL_15:
        v13 = v12 - v32;
        if ( *(_QWORD *)(a3 + 24) - result <= 0x10 )
        {
          sub_CB6200(a3, ", at location [SP", 0x11u);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_42EAC70);
          *(_BYTE *)(result + 16) = 80;
          *(__m128i *)result = si128;
          *(_QWORD *)(a3 + 32) += 17LL;
        }
        if ( v13 > 0 )
        {
          v29 = *(_BYTE **)(a3 + 32);
          if ( *(_BYTE **)(a3 + 24) == v29 )
          {
            v30 = sub_CB6200(a3, (unsigned __int8 *)"+", 1u);
            sub_CB59F0(v30, v13);
          }
          else
          {
            *v29 = 43;
            ++*(_QWORD *)(a3 + 32);
            sub_CB59F0(a3, v13);
          }
          v15 = *(_BYTE **)(a3 + 32);
          if ( *(_BYTE **)(a3 + 24) != v15 )
          {
LABEL_21:
            *v15 = 93;
            result = *(_QWORD *)(a3 + 32) + 1LL;
            *(_QWORD *)(a3 + 32) = result;
            goto LABEL_22;
          }
        }
        else
        {
          if ( v13 )
            sub_CB59F0(a3, v13);
          v15 = *(_BYTE **)(a3 + 32);
          if ( *(_BYTE **)(a3 + 24) != v15 )
            goto LABEL_21;
        }
        sub_CB6200(a3, (unsigned __int8 *)"]", 1u);
        result = *(_QWORD *)(a3 + 32);
LABEL_22:
        if ( result != *(_QWORD *)(a3 + 24) )
        {
          *(_BYTE *)result = 10;
          ++*(_QWORD *)(a3 + 32);
          goto LABEL_24;
        }
        ++v9;
        result = sub_CB6200(a3, (unsigned __int8 *)"\n", 1u);
        if ( v9 == v33 )
          return result;
LABEL_25:
        result = *(_QWORD *)(a1 + 8);
      }
    }
  }
  return result;
}
