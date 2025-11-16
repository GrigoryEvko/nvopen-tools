// Function: sub_15EA640
// Address: 0x15ea640
//
_BYTE *sub_15EA640()
{
  _BYTE *v0; // rax
  _BYTE *v1; // r12
  __int64 *v2; // rbx
  _BYTE v4[16]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v5; // [rsp+20h] [rbp-30h]

  v0 = (_BYTE *)sub_22077B0(28208);
  v1 = v0;
  if ( v0 )
  {
    memset(v0, 0, 0x6E30u);
    v1[8] = 1;
    v2 = (__int64 *)(v1 + 16);
    v4[0] = 0;
    v5 = 0;
    do
    {
      if ( v2 )
      {
        *v2 = (__int64)(v2 + 2);
        sub_15EA590(v2, v4, (__int64)v4);
        *((_WORD *)v2 + 16) = v5;
      }
      v2 += 6;
    }
    while ( v2 != (__int64 *)(v1 + 1552) );
    *((_QWORD *)v1 + 3524) = 0;
    *((_QWORD *)v1 + 194) = v1 + 1568;
    *((_QWORD *)v1 + 195) = 0x2000000000LL;
    *((_QWORD *)v1 + 3525) = 0;
  }
  return v1;
}
