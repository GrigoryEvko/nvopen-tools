// Function: sub_73F0A0
// Address: 0x73f0a0
//
_QWORD *__fastcall sub_73F0A0(__m128i *a1, __int64 a2)
{
  __m128i *v2; // rdi
  __int64 v3; // rax
  _QWORD *v4; // rcx
  __int64 v5; // r12
  _QWORD *v7; // r12
  __int64 v8; // [rsp+0h] [rbp-20h] BYREF
  __m128i *v9; // [rsp+8h] [rbp-18h] BYREF

  v9 = a1;
  v8 = a2;
  sub_73EE70(&v8, &v9);
  v2 = v9;
  v3 = v9[7].m128i_i64[1];
  if ( v3 )
  {
    v4 = 0;
    while ( 1 )
    {
      if ( *(_BYTE *)(v3 + 16) == 3 )
      {
        v5 = *(_QWORD *)(v3 + 8);
        if ( v8 == *(_QWORD *)(v5 + 160) )
          break;
      }
      v4 = (_QWORD *)v3;
      if ( !*(_QWORD *)v3 )
        goto LABEL_10;
      v3 = *(_QWORD *)v3;
    }
    if ( v4 )
    {
      *v4 = *(_QWORD *)v3;
      *(_QWORD *)v3 = v2[7].m128i_i64[1];
      v2[7].m128i_i64[1] = v3;
    }
    return (_QWORD *)v5;
  }
  else
  {
LABEL_10:
    v7 = sub_7259C0(13);
    v7[21] = v9;
    v7[20] = v8;
    sub_8D6090(v7);
    sub_728520(v9, 3, (__int64)v7);
    return v7;
  }
}
