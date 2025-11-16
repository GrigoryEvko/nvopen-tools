// Function: sub_2097DA0
// Address: 0x2097da0
//
void __fastcall sub_2097DA0(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // r15
  __int64 v4; // r13
  __int64 v5; // r12
  int v6; // ebx
  __int64 v7; // rax
  __int64 v8; // r8
  unsigned int v9; // r14d
  char *v10; // rax
  char *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // rdi
  _BYTE *v15; // rax
  __int64 v16; // rsi
  _WORD *v17; // rdx
  _BYTE *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // [rsp+10h] [rbp-70h]
  __int64 v22; // [rsp+10h] [rbp-70h]
  __int64 v23; // [rsp+10h] [rbp-70h]
  int v25; // [rsp+24h] [rbp-5Ch]
  __m128i v27; // [rsp+30h] [rbp-50h] BYREF
  _QWORD v28[8]; // [rsp+40h] [rbp-40h] BYREF

  v3 = a1;
  v4 = a2;
  sub_2097C50(a1, a2, a3);
  v25 = *(_DWORD *)(a1 + 56);
  if ( v25 )
  {
    v5 = 0;
    v6 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v10 = *(char **)(a2 + 16);
        v11 = *(char **)(a2 + 24);
        if ( v6 )
        {
          if ( (unsigned __int64)(v10 - v11) <= 1 )
          {
            sub_16E7EE0(a2, ", ", 2u);
          }
          else
          {
            *(_WORD *)v11 = 8236;
            *(_QWORD *)(a2 + 24) += 2LL;
          }
        }
        else if ( v11 == v10 )
        {
          sub_16E7EE0(a2, " ", 1u);
        }
        else
        {
          *v11 = 32;
          ++*(_QWORD *)(a2 + 24);
        }
        v7 = v5 + *(_QWORD *)(a1 + 32);
        v8 = *(_QWORD *)v7;
        if ( *(_QWORD *)v7 )
          break;
        v20 = *(_QWORD *)(a2 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v20) <= 5 )
        {
          sub_16E7EE0(a2, "<null>", 6u);
        }
        else
        {
          *(_DWORD *)v20 = 1819635260;
          *(_WORD *)(v20 + 4) = 15980;
          *(_QWORD *)(a2 + 24) += 6LL;
        }
LABEL_11:
        ++v6;
        v5 += 40;
        if ( v6 == v25 )
          goto LABEL_20;
      }
      if ( *(_WORD *)(v8 + 24) == 1 || *(_DWORD *)(v8 + 56) )
      {
        v9 = *(_DWORD *)(v7 + 8);
        v27.m128i_i64[0] = *(_QWORD *)v7;
        v28[0] = sub_2094190;
        v28[1] = sub_2094180;
        sub_2094180(v27.m128i_i64, a2);
        if ( v28[0] )
          ((void (__fastcall *)(__m128i *, __m128i *, __int64))v28[0])(&v27, &v27, 3);
        if ( v9 )
        {
          v18 = *(_BYTE **)(a2 + 24);
          if ( (unsigned __int64)v18 >= *(_QWORD *)(a2 + 16) )
          {
            v19 = sub_16E7DE0(a2, 58);
          }
          else
          {
            v19 = a2;
            *(_QWORD *)(a2 + 24) = v18 + 1;
            *v18 = 58;
          }
          sub_16E7A90(v19, v9);
        }
        goto LABEL_11;
      }
      v21 = *(_QWORD *)v7;
      sub_2095B00(&v27, *(_QWORD *)v7, a3);
      v12 = sub_16E7EE0(a2, (char *)v27.m128i_i64[0], v27.m128i_u64[1]);
      v13 = v21;
      v14 = v12;
      v15 = *(_BYTE **)(v12 + 24);
      if ( (unsigned __int64)v15 >= *(_QWORD *)(v14 + 16) )
      {
        sub_16E7DE0(v14, 58);
        v13 = v21;
      }
      else
      {
        *(_QWORD *)(v14 + 24) = v15 + 1;
        *v15 = 58;
      }
      if ( (_QWORD *)v27.m128i_i64[0] != v28 )
      {
        v22 = v13;
        j_j___libc_free_0(v27.m128i_i64[0], v28[0] + 1LL);
        v13 = v22;
      }
      v23 = v13;
      ++v6;
      v5 += 40;
      sub_2094480(v13, a2);
      sub_20945B0(v23, a2, (__int64)a3);
      if ( v6 == v25 )
      {
LABEL_20:
        v4 = a2;
        v3 = a1;
        break;
      }
    }
  }
  v16 = *(_QWORD *)(v3 + 72);
  v27.m128i_i64[0] = v16;
  if ( v16 )
  {
    sub_1623A60((__int64)&v27, v16, 2);
    if ( v27.m128i_i64[0] )
    {
      v17 = *(_WORD **)(v4 + 24);
      if ( *(_QWORD *)(v4 + 16) - (_QWORD)v17 <= 1u )
      {
        sub_16E7EE0(v4, ", ", 2u);
      }
      else
      {
        *v17 = 8236;
        *(_QWORD *)(v4 + 24) += 2LL;
      }
      sub_15C7170(&v27, v4);
      if ( v27.m128i_i64[0] )
        sub_161E7C0((__int64)&v27, v27.m128i_i64[0]);
    }
  }
}
