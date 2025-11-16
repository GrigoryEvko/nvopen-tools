// Function: sub_1843600
// Address: 0x1843600
//
void __fastcall sub_1843600(_QWORD *a1, __m128i *a2)
{
  _QWORD *v4; // rax
  _QWORD *v5; // r8
  unsigned __int64 v6; // rsi
  _QWORD *v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // rdx
  _QWORD *v10; // rbx
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rdx
  bool v13; // zf
  unsigned __int32 v14; // eax
  _QWORD *v15; // rax
  char v16; // si
  _QWORD *v17; // r15
  __int64 v18; // rax
  _BOOL4 v19; // ebx
  __m128i *v20; // rax
  unsigned __int32 v21; // eax
  unsigned __int64 v22; // rax
  unsigned __int32 v23; // eax

  v4 = (_QWORD *)a1[14];
  if ( !v4 )
    goto LABEL_8;
  v5 = a1 + 13;
  v6 = a2->m128i_i64[0];
  v7 = a1 + 13;
  do
  {
    while ( 1 )
    {
      v8 = v4[2];
      v9 = v4[3];
      if ( v4[4] >= v6 )
        break;
      v4 = (_QWORD *)v4[3];
      if ( !v9 )
        goto LABEL_6;
    }
    v7 = v4;
    v4 = (_QWORD *)v4[2];
  }
  while ( v8 );
LABEL_6:
  if ( v5 == v7 || v7[4] > v6 )
  {
LABEL_8:
    v10 = (_QWORD *)a1[8];
    if ( v10 )
    {
      v11 = a2->m128i_i64[0];
      v12 = v10[4];
      v13 = v12 == a2->m128i_i64[0];
      if ( v12 <= a2->m128i_i64[0] )
        goto LABEL_10;
LABEL_15:
      while ( 1 )
      {
        v15 = (_QWORD *)v10[2];
        v16 = 1;
        if ( !v15 )
          break;
        while ( 1 )
        {
          v10 = v15;
          v12 = v15[4];
          v13 = v12 == v11;
          if ( v12 > v11 )
            break;
LABEL_10:
          if ( v13 )
          {
            v14 = *((_DWORD *)v10 + 10);
            if ( a2->m128i_i32[2] < v14 || a2->m128i_i32[2] == v14 && a2->m128i_i8[12] < *((_BYTE *)v10 + 44) )
              goto LABEL_15;
          }
          v15 = (_QWORD *)v10[3];
          v16 = 0;
          if ( !v15 )
            goto LABEL_16;
        }
      }
LABEL_16:
      v17 = v10;
      if ( !v16 )
        goto LABEL_19;
    }
    else
    {
      v10 = a1 + 7;
    }
    if ( v10 == (_QWORD *)a1[9] )
    {
      v17 = v10;
LABEL_20:
      v19 = 1;
      if ( a1 + 7 != v17 )
      {
        v22 = v17[4];
        v19 = 1;
        if ( a2->m128i_i64[0] >= v22 )
        {
          if ( a2->m128i_i64[0] != v22
            || (v23 = *((_DWORD *)v17 + 10), a2->m128i_i32[2] >= v23)
            && (a2->m128i_i32[2] != v23 || a2->m128i_i8[12] >= *((_BYTE *)v17 + 44)) )
          {
            v19 = 0;
          }
        }
      }
      v20 = (__m128i *)sub_22077B0(48);
      v20[2] = _mm_loadu_si128(a2);
      sub_220F040(v19, v20, v17, a1 + 7);
      ++a1[11];
      sub_1843480(a1, a2->m128i_i64);
      return;
    }
    v17 = v10;
    v18 = sub_220EF80(v10);
    v11 = a2->m128i_i64[0];
    v12 = *(_QWORD *)(v18 + 32);
    v10 = (_QWORD *)v18;
LABEL_19:
    if ( v11 <= v12 )
    {
      if ( v11 != v12 )
        return;
      v21 = a2->m128i_u32[2];
      if ( *((_DWORD *)v10 + 10) >= v21 && (*((_DWORD *)v10 + 10) != v21 || *((_BYTE *)v10 + 44) >= a2->m128i_i8[12]) )
        return;
      if ( !v17 )
        return;
    }
    goto LABEL_20;
  }
}
