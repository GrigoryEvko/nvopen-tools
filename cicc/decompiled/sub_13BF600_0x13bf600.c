// Function: sub_13BF600
// Address: 0x13bf600
//
__int64 __fastcall sub_13BF600(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r12
  __m128i *v4; // rdx
  __m128i si128; // xmm0
  __int64 v6; // rdi
  __int64 v7; // rdx
  _BYTE *v8; // r14
  __int64 v9; // r13
  __int64 i; // rbx
  __int64 v11; // r10
  void *v12; // rdx
  void *v13; // rdx
  __int64 v14; // [rsp+0h] [rbp-40h]
  __int64 v15; // [rsp+8h] [rbp-38h]

  result = a1 + 8;
  v3 = *(_QWORD *)(a1 + 24);
  v14 = a1 + 8;
  if ( v3 != a1 + 8 )
  {
    do
    {
      v4 = *(__m128i **)(a2 + 24);
      if ( *(_QWORD *)(a2 + 16) - (_QWORD)v4 <= 0x14u )
      {
        sub_16E7EE0(a2, "  DomFrontier for BB ", 21);
        v6 = *(_QWORD *)(v3 + 32);
        if ( !v6 )
        {
LABEL_24:
          v13 = *(void **)(a2 + 24);
          if ( *(_QWORD *)(a2 + 16) - (_QWORD)v13 <= 0xDu )
          {
            sub_16E7EE0(a2, " <<exit node>>", 14);
            v7 = *(_QWORD *)(a2 + 24);
          }
          else
          {
            qmemcpy(v13, " <<exit node>>", 14);
            v7 = *(_QWORD *)(a2 + 24) + 14LL;
            *(_QWORD *)(a2 + 24) = v7;
          }
          goto LABEL_5;
        }
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_4289C50);
        v4[1].m128i_i32[0] = 1111629938;
        v4[1].m128i_i8[4] = 32;
        *v4 = si128;
        *(_QWORD *)(a2 + 24) += 21LL;
        v6 = *(_QWORD *)(v3 + 32);
        if ( !v6 )
          goto LABEL_24;
      }
      sub_15537D0(v6, a2, 0);
      v7 = *(_QWORD *)(a2 + 24);
LABEL_5:
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v7) <= 4 )
      {
        sub_16E7EE0(a2, " is:\t", 5);
        v8 = *(_BYTE **)(a2 + 24);
      }
      else
      {
        *(_DWORD *)v7 = 980642080;
        *(_BYTE *)(v7 + 4) = 9;
        v8 = (_BYTE *)(*(_QWORD *)(a2 + 24) + 5LL);
        *(_QWORD *)(a2 + 24) = v8;
      }
      v9 = *(_QWORD *)(v3 + 64);
      for ( i = v3 + 48; i != v9; v9 = sub_220EF30(v9) )
      {
        while ( 1 )
        {
          v11 = *(_QWORD *)(v9 + 32);
          if ( (unsigned __int64)v8 >= *(_QWORD *)(a2 + 16) )
            break;
          *(_QWORD *)(a2 + 24) = v8 + 1;
          *v8 = 32;
          if ( !v11 )
            goto LABEL_14;
LABEL_10:
          sub_15537D0(v11, a2, 0);
          v8 = *(_BYTE **)(a2 + 24);
LABEL_11:
          v9 = sub_220EF30(v9);
          if ( i == v9 )
            goto LABEL_16;
        }
        v15 = *(_QWORD *)(v9 + 32);
        sub_16E7DE0(a2, 32);
        v11 = v15;
        if ( v15 )
          goto LABEL_10;
LABEL_14:
        v12 = *(void **)(a2 + 24);
        if ( *(_QWORD *)(a2 + 16) - (_QWORD)v12 <= 0xCu )
        {
          sub_16E7EE0(a2, "<<exit node>>", 13);
          v8 = *(_BYTE **)(a2 + 24);
          goto LABEL_11;
        }
        qmemcpy(v12, "<<exit node>>", 13);
        v8 = (_BYTE *)(*(_QWORD *)(a2 + 24) + 13LL);
        *(_QWORD *)(a2 + 24) = v8;
      }
LABEL_16:
      if ( (unsigned __int64)v8 >= *(_QWORD *)(a2 + 16) )
      {
        sub_16E7DE0(a2, 10);
      }
      else
      {
        *(_QWORD *)(a2 + 24) = v8 + 1;
        *v8 = 10;
      }
      result = sub_220EF30(v3);
      v3 = result;
    }
    while ( v14 != result );
  }
  return result;
}
