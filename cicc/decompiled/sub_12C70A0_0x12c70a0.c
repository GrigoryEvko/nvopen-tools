// Function: sub_12C70A0
// Address: 0x12c70a0
//
__int64 *__fastcall sub_12C70A0(__int64 *a1, __int64 a2)
{
  _BYTE *v3; // rdi
  _BYTE *v4; // r14
  size_t v5; // r13
  __int64 v7; // rax
  size_t v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = a1 + 2;
  v4 = *(_BYTE **)a2;
  if ( *(_QWORD *)a2 )
  {
    v5 = *(_QWORD *)(a2 + 8);
    *a1 = (__int64)v3;
    v8[0] = v5;
    if ( v5 > 0xF )
    {
      v7 = sub_22409D0(a1, v8, 0);
      *a1 = v7;
      v3 = (_BYTE *)v7;
      a1[2] = v8[0];
    }
    else
    {
      if ( v5 == 1 )
      {
        *((_BYTE *)a1 + 16) = *v4;
LABEL_5:
        a1[1] = v5;
        v3[v5] = 0;
        return a1;
      }
      if ( !v5 )
        goto LABEL_5;
    }
    memcpy(v3, v4, v5);
    v5 = v8[0];
    v3 = (_BYTE *)*a1;
    goto LABEL_5;
  }
  *a1 = (__int64)v3;
  a1[1] = 0;
  *((_BYTE *)a1 + 16) = 0;
  return a1;
}
