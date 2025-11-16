// Function: sub_1DD8D40
// Address: 0x1dd8d40
//
char *__fastcall sub_1DD8D40(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  _BYTE *v4; // rsi
  __int64 v5; // rdi
  __int64 v7; // [rsp+8h] [rbp-18h] BYREF

  v7 = a2;
  v3 = *(_QWORD *)(a1 + 112);
  if ( v3 != *(_QWORD *)(a1 + 120) )
    *(_QWORD *)(a1 + 120) = v3;
  v4 = *(_BYTE **)(a1 + 96);
  if ( v4 == *(_BYTE **)(a1 + 104) )
  {
    sub_1D4AF10(a1 + 88, v4, &v7);
    return sub_1DD8D00(v7, (char *)a1);
  }
  else
  {
    v5 = v7;
    if ( v4 )
    {
      *(_QWORD *)v4 = v7;
      v4 = *(_BYTE **)(a1 + 96);
      v5 = v7;
    }
    *(_QWORD *)(a1 + 96) = v4 + 8;
    return sub_1DD8D00(v5, (char *)a1);
  }
}
