// Function: sub_2CE4DC0
// Address: 0x2ce4dc0
//
__int64 __fastcall sub_2CE4DC0(_QWORD *a1, const __m128i *a2)
{
  const __m128i *v2; // rbp
  _QWORD *v3; // rax
  _QWORD *v6; // r10
  unsigned __int64 v7; // rdi
  _QWORD *v8; // rsi
  _QWORD *v9; // rdx
  unsigned __int64 v10; // rdx
  bool v11; // cl
  const __m128i *v13[2]; // [rsp-10h] [rbp-10h] BYREF

  v3 = (_QWORD *)a1[2];
  v6 = a1 + 1;
  if ( !v3 )
  {
    v8 = a1 + 1;
    goto LABEL_12;
  }
  v7 = a2->m128i_i64[0];
  v8 = v6;
  while ( 1 )
  {
    v10 = v3[4];
    v11 = v10 < v7;
    if ( v10 == v7 )
      v11 = v3[5] < a2->m128i_i64[1];
    v9 = (_QWORD *)v3[3];
    if ( !v11 )
    {
      v9 = (_QWORD *)v3[2];
      v8 = v3;
    }
    if ( !v9 )
      break;
    v3 = v9;
  }
  if ( v6 == v8 )
    goto LABEL_12;
  if ( v8[4] == v7 )
  {
    if ( a2->m128i_i64[1] < v8[5] )
      goto LABEL_12;
  }
  else if ( v8[4] > v7 )
  {
LABEL_12:
    v13[1] = v2;
    v13[0] = a2;
    return sub_2CE4CC0(a1, v8, v13) + 48;
  }
  return (__int64)(v8 + 6);
}
