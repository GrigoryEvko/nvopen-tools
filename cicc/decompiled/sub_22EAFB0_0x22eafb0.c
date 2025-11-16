// Function: sub_22EAFB0
// Address: 0x22eafb0
//
__int64 __fastcall sub_22EAFB0(__int64 a1, char *a2)
{
  __int64 v2; // rcx
  char v3; // al
  unsigned __int64 v4; // rdx
  __int64 v5; // r12
  _BYTE *v6; // rax
  _WORD *v8; // rdx
  __m128i si128; // xmm0

  v2 = *(_QWORD *)(a1 + 32);
  v3 = *a2;
  v4 = *(_QWORD *)(a1 + 24) - v2;
  if ( *a2 )
  {
    if ( v3 == 1 )
    {
      if ( v4 > 4 )
      {
        *(_DWORD *)v2 = 1701080693;
        v5 = a1;
        *(_BYTE *)(v2 + 4) = 102;
        *(_QWORD *)(a1 + 32) += 5LL;
        return v5;
      }
      return sub_CB6200(a1, (unsigned __int8 *)"undef", 5u);
    }
    else
    {
      if ( v3 != 6 )
      {
        if ( v3 != 3 )
        {
          if ( v3 == 5 )
          {
            if ( v4 <= 0x1A )
            {
              v5 = sub_CB6200(a1, "constantrange incl. undef <", 0x1Bu);
            }
            else
            {
              si128 = _mm_load_si128((const __m128i *)&xmmword_4366C50);
              v5 = a1;
              qmemcpy((void *)(v2 + 16), "cl. undef <", 11);
              *(__m128i *)v2 = si128;
              *(_QWORD *)(a1 + 32) += 27LL;
            }
          }
          else
          {
            if ( v3 != 4 )
            {
              if ( v4 <= 8 )
              {
                v5 = sub_CB6200(a1, (unsigned __int8 *)"constant<", 9u);
              }
              else
              {
                *(_BYTE *)(v2 + 8) = 60;
                v5 = a1;
                *(_QWORD *)v2 = 0x746E6174736E6F63LL;
                *(_QWORD *)(a1 + 32) += 9LL;
              }
              goto LABEL_9;
            }
            if ( v4 <= 0xD )
            {
              v5 = sub_CB6200(a1, "constantrange<", 0xEu);
            }
            else
            {
              v5 = a1;
              qmemcpy((void *)v2, "constantrange<", 14);
              *(_QWORD *)(a1 + 32) += 14LL;
            }
          }
          sub_C49420((__int64)(a2 + 8), v5, 1);
          v8 = *(_WORD **)(v5 + 32);
          if ( *(_QWORD *)(v5 + 24) - (_QWORD)v8 <= 1u )
          {
            v5 = sub_CB6200(v5, (unsigned __int8 *)", ", 2u);
          }
          else
          {
            *v8 = 8236;
            *(_QWORD *)(v5 + 32) += 2LL;
          }
          sub_C49420((__int64)(a2 + 24), v5, 1);
          v6 = *(_BYTE **)(v5 + 32);
          if ( *(_BYTE **)(v5 + 24) == v6 )
            return sub_CB6200(v5, (unsigned __int8 *)">", 1u);
LABEL_21:
          *v6 = 62;
          ++*(_QWORD *)(v5 + 32);
          return v5;
        }
        if ( v4 <= 0xB )
        {
          v5 = sub_CB6200(a1, "notconstant<", 0xCu);
        }
        else
        {
          v5 = a1;
          qmemcpy((void *)v2, "notconstant<", 12);
          *(_QWORD *)(a1 + 32) += 12LL;
        }
LABEL_9:
        sub_A69870(*((_QWORD *)a2 + 1), (_BYTE *)v5, 0);
        v6 = *(_BYTE **)(v5 + 32);
        if ( *(_BYTE **)(v5 + 24) == v6 )
          return sub_CB6200(v5, (unsigned __int8 *)">", 1u);
        goto LABEL_21;
      }
      if ( v4 > 0xA )
      {
        v5 = a1;
        qmemcpy((void *)v2, "overdefined", 11);
        *(_QWORD *)(a1 + 32) += 11LL;
        return v5;
      }
      return sub_CB6200(a1, "overdefined", 0xBu);
    }
  }
  else
  {
    if ( v4 > 6 )
    {
      *(_DWORD *)v2 = 1852534389;
      v5 = a1;
      *(_WORD *)(v2 + 4) = 30575;
      *(_BYTE *)(v2 + 6) = 110;
      *(_QWORD *)(a1 + 32) += 7LL;
      return v5;
    }
    return sub_CB6200(a1, (unsigned __int8 *)"unknown", 7u);
  }
}
