// Function: sub_384CCA0
// Address: 0x384cca0
//
__int64 __fastcall sub_384CCA0(__int64 a1, __int64 a2)
{
  __int64 **v2; // r12
  __int64 **v3; // rbx
  __int64 v5; // r14
  char v6; // r15
  __int64 v7; // rdi
  __m128i *v8; // rdx
  __m128i si128; // xmm0
  char v11; // r15
  char v12; // [rsp+Fh] [rbp-31h]

  v2 = *(__int64 ***)(a2 + 24);
  v3 = *(__int64 ***)(a2 + 16);
  v12 = 0;
  if ( v2 != v3 )
  {
    while ( 1 )
    {
      v5 = **v3;
      if ( v5 )
        break;
      v6 = sub_160E740();
      if ( v6 )
      {
        if ( !v12 )
          sub_16E7EE0(*(_QWORD *)(a1 + 192), *(char **)(a1 + 160), *(_QWORD *)(a1 + 168));
        v7 = *(_QWORD *)(a1 + 192);
        v8 = *(__m128i **)(v7 + 24);
        if ( *(_QWORD *)(v7 + 16) - (_QWORD)v8 <= 0x19u )
        {
          sub_16E7EE0(v7, "\nPrinting <null> Function\n", 0x1Au);
          v12 = v6;
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_44CB540);
          v12 = v6;
          qmemcpy(&v8[1], " Function\n", 10);
          *v8 = si128;
          *(_QWORD *)(v7 + 24) += 26LL;
        }
        if ( v2 == ++v3 )
          return 0;
      }
      else
      {
LABEL_4:
        if ( v2 == ++v3 )
          return 0;
      }
    }
    if ( !sub_15E4F60(**v3) )
    {
      sub_1649960(v5);
      v11 = sub_160E740();
      if ( v11 )
      {
        if ( !v12 )
          sub_16E7EE0(*(_QWORD *)(a1 + 192), *(char **)(a1 + 160), *(_QWORD *)(a1 + 168));
        sub_1559E80(v5, *(_QWORD *)(a1 + 192), 0, 0, 0);
        v12 = v11;
      }
    }
    goto LABEL_4;
  }
  return 0;
}
