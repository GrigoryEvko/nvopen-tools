// Function: sub_25B0490
// Address: 0x25b0490
//
void __fastcall sub_25B0490(_QWORD *a1, __m128i *a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // rcx
  unsigned __int64 v5; // rdx
  unsigned __int32 v6; // eax
  __int64 v7; // rax
  char v8; // si
  __int64 v9; // r15
  __int64 v10; // rax
  char v11; // bl
  __m128i *v12; // rax
  unsigned __int32 v13; // eax
  unsigned __int64 v14; // rax
  unsigned __int32 v15; // eax

  if ( !(unsigned __int8)sub_25AFE60(a1, a2->m128i_i64) )
  {
    v3 = a1[8];
    if ( v3 )
    {
      v4 = a2->m128i_i64[0];
      while ( 1 )
      {
        v5 = *(_QWORD *)(v3 + 32);
        if ( v5 > v4
          || v5 == v4
          && ((v6 = *(_DWORD *)(v3 + 40), a2->m128i_i32[2] < v6)
           || a2->m128i_i32[2] == v6 && a2->m128i_i8[12] < *(_BYTE *)(v3 + 44)) )
        {
          v7 = *(_QWORD *)(v3 + 16);
          v8 = 1;
          if ( !v7 )
            goto LABEL_12;
        }
        else
        {
          v7 = *(_QWORD *)(v3 + 24);
          v8 = 0;
          if ( !v7 )
          {
LABEL_12:
            v9 = v3;
            if ( v8 )
              goto LABEL_13;
LABEL_15:
            if ( v4 > v5 )
              goto LABEL_16;
            if ( v4 == v5 )
            {
              v13 = a2->m128i_u32[2];
              if ( *(_DWORD *)(v3 + 40) < v13 || *(_DWORD *)(v3 + 40) == v13 && *(_BYTE *)(v3 + 44) < a2->m128i_i8[12] )
              {
                if ( v9 )
                  goto LABEL_16;
              }
            }
            goto LABEL_18;
          }
        }
        v3 = v7;
      }
    }
    v3 = (__int64)(a1 + 7);
LABEL_13:
    if ( a1[9] != v3 )
    {
      v9 = v3;
      v10 = sub_220EF80(v3);
      v4 = a2->m128i_i64[0];
      v5 = *(_QWORD *)(v10 + 32);
      v3 = v10;
      goto LABEL_15;
    }
    v9 = v3;
LABEL_16:
    v11 = 1;
    if ( a1 + 7 != (_QWORD *)v9 )
    {
      v14 = *(_QWORD *)(v9 + 32);
      v11 = 1;
      if ( a2->m128i_i64[0] >= v14 )
      {
        if ( a2->m128i_i64[0] != v14
          || (v15 = *(_DWORD *)(v9 + 40), a2->m128i_i32[2] >= v15)
          && (a2->m128i_i32[2] != v15 || a2->m128i_i8[12] >= *(_BYTE *)(v9 + 44)) )
        {
          v11 = 0;
        }
      }
    }
    v12 = (__m128i *)sub_22077B0(0x30u);
    v12[2] = _mm_loadu_si128(a2);
    sub_220F040(v11, (__int64)v12, (_QWORD *)v9, a1 + 7);
    ++a1[11];
LABEL_18:
    sub_25B0310(a1, a2->m128i_i64);
  }
}
