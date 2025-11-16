// Function: sub_7CB020
// Address: 0x7cb020
//
__int64 __fastcall sub_7CB020(_QWORD *a1)
{
  __int64 v2; // r14
  __int64 v3; // rbx
  __int64 v4; // rdi
  _BYTE *v5; // r15
  _BYTE *v6; // rax
  _BYTE *v7; // rdx
  _BYTE *v8; // r13
  _BYTE *v9; // rsi
  __int64 result; // rax
  __int64 v11; // [rsp+8h] [rbp-38h]

  v2 = a1[1];
  if ( v2 <= 1 )
  {
    v4 = 2;
    v3 = 2;
  }
  else
  {
    v3 = v2 + (v2 >> 1) + 1;
    v4 = v3;
  }
  v5 = (_BYTE *)*a1;
  v11 = a1[2];
  v6 = (_BYTE *)sub_823970(v4);
  v7 = v5;
  v8 = v6;
  v9 = &v6[v11];
  if ( v11 > 0 )
  {
    do
    {
      if ( v6 )
        *v6 = *v7;
      ++v6;
      ++v7;
    }
    while ( v6 != v9 );
  }
  result = sub_823A00(v5, v2);
  *a1 = v8;
  a1[1] = v3;
  return result;
}
