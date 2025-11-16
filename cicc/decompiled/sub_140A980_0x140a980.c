// Function: sub_140A980
// Address: 0x140a980
//
__m128i *__fastcall sub_140A980(__m128i *a1, __int64 a2, unsigned __int8 a3, _QWORD *a4)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  char *v9; // rbx
  int v10; // ecx
  char *v11; // rdx
  __int64 v12; // r14
  int v13; // r15d
  __int64 v14; // r13
  __m128i v15; // xmm0
  __int64 v16; // r13
  __int64 v17; // r13
  int v18; // [rsp+Ch] [rbp-44h]
  int v19[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v6 = sub_1649960(a2);
  if ( a4
    && (unsigned __int8)sub_149B630(*a4, v6, v7, v19)
    && (((int)*(unsigned __int8 *)(*a4 + v19[0] / 4) >> (2 * (v19[0] & 3))) & 3) != 0 )
  {
    v9 = (char *)&unk_428B7C0;
    v10 = 285;
    while ( v19[0] != v10 )
    {
      if ( v19[0] == *((_DWORD *)v9 + 5) )
      {
        v9 += 20;
        break;
      }
      if ( v19[0] == *((_DWORD *)v9 + 10) )
      {
        v9 += 40;
        break;
      }
      if ( v19[0] == *((_DWORD *)v9 + 15) )
      {
        v9 += 60;
        break;
      }
      if ( v9 + 80 == (char *)&unk_428B9F0 )
      {
        if ( v19[0] == *((_DWORD *)v9 + 20) )
        {
          v9 = (char *)&unk_428B9F0;
          goto LABEL_18;
        }
        v11 = v9 + 100;
        if ( v19[0] != *((_DWORD *)v9 + 25) )
        {
          v11 = v9 + 120;
          if ( v19[0] != *((_DWORD *)v9 + 30) )
            goto LABEL_3;
        }
        v9 = v11;
        break;
      }
      v10 = *((_DWORD *)v9 + 20);
      v9 += 80;
    }
    if ( v9 == (char *)jpt_14126B9 )
      goto LABEL_3;
LABEL_18:
    if ( v9[4] == ((unsigned __int8)v9[4] & a3) )
    {
      v12 = *(_QWORD *)(a2 + 24);
      v13 = *((_DWORD *)v9 + 3);
      v18 = *((_DWORD *)v9 + 4);
      v14 = **(_QWORD **)(v12 + 16);
      if ( v14 == sub_16471D0(*(_QWORD *)v12, 0) && *((_DWORD *)v9 + 2) == *(_DWORD *)(v12 + 12) - 1 )
      {
        if ( v13 < 0
          || (v17 = (unsigned int)(v13 + 1),
              (unsigned __int8)sub_1642F90(*(_QWORD *)(*(_QWORD *)(v12 + 16) + 8 * v17), 32))
          || (unsigned __int8)sub_1642F90(*(_QWORD *)(*(_QWORD *)(v12 + 16) + 8 * v17), 64) )
        {
          if ( v18 < 0
            || (v16 = (unsigned int)(v18 + 1),
                (unsigned __int8)sub_1642F90(*(_QWORD *)(*(_QWORD *)(v12 + 16) + 8 * v16), 32))
            || (unsigned __int8)sub_1642F90(*(_QWORD *)(*(_QWORD *)(v12 + 16) + 8 * v16), 64) )
          {
            v15 = _mm_loadu_si128((const __m128i *)(v9 + 4));
            a1[1].m128i_i8[0] = 1;
            *a1 = v15;
            return a1;
          }
        }
      }
    }
  }
LABEL_3:
  a1[1].m128i_i8[0] = 0;
  return a1;
}
