// Function: sub_1E2D790
// Address: 0x1e2d790
//
void __fastcall sub_1E2D790(_QWORD *a1, __int64 a2)
{
  _BYTE *v2; // r9
  __int64 v3; // rsi
  unsigned int v4; // edx
  __int64 v5; // rax
  __int64 v6; // [rsp+8h] [rbp-8h] BYREF

  v2 = (_BYTE *)a1[214];
  v6 = a2;
  v3 = a1[213];
  if ( (__int64)&v2[-v3] >> 3 )
  {
    v4 = 0;
    v5 = 0;
    while ( *(_QWORD *)(v3 + 8 * v5) != v6 )
    {
      v5 = ++v4;
      if ( v4 >= (unsigned __int64)((__int64)&v2[-v3] >> 3) )
        goto LABEL_6;
    }
  }
  else
  {
LABEL_6:
    if ( (_BYTE *)a1[215] == v2 )
    {
      sub_1E2D600((__int64)(a1 + 213), v2, &v6);
    }
    else
    {
      if ( v2 )
      {
        *(_QWORD *)v2 = v6;
        v2 = (_BYTE *)a1[214];
      }
      a1[214] = v2 + 8;
    }
  }
}
