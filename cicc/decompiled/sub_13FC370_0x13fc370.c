// Function: sub_13FC370
// Address: 0x13fc370
//
__int64 __fastcall sub_13FC370(__int64 a1)
{
  _BYTE *v2; // rdi
  _BYTE *v3; // r13
  _BYTE *v4; // r12
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v7; // rax
  unsigned int v8; // r12d
  _BYTE *v10; // [rsp+0h] [rbp-60h] BYREF
  __int64 v11; // [rsp+8h] [rbp-58h]
  _BYTE v12[80]; // [rsp+10h] [rbp-50h] BYREF

  v11 = 0x400000000LL;
  v10 = v12;
  sub_13F9EC0(a1, (__int64)&v10);
  v2 = v10;
  v3 = &v10[8 * (unsigned int)v11];
  if ( v3 == v10 )
  {
    v8 = 1;
  }
  else
  {
    v4 = v10;
    v5 = a1 + 56;
    while ( 1 )
    {
      while ( 1 )
      {
        v6 = *(_QWORD *)(*(_QWORD *)v4 + 8LL);
        if ( v6 )
          break;
LABEL_6:
        v4 += 8;
        if ( v3 == v4 )
          goto LABEL_7;
      }
      while ( 1 )
      {
        v7 = sub_1648700(v6);
        if ( (unsigned __int8)(*(_BYTE *)(v7 + 16) - 25) <= 9u )
          break;
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          goto LABEL_6;
      }
LABEL_9:
      if ( !sub_1377F70(v5, *(_QWORD *)(v7 + 40)) )
        break;
      while ( 1 )
      {
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          break;
        v7 = sub_1648700(v6);
        if ( (unsigned __int8)(*(_BYTE *)(v7 + 16) - 25) <= 9u )
          goto LABEL_9;
      }
      v4 += 8;
      if ( v3 == v4 )
      {
LABEL_7:
        v2 = v10;
        v8 = 1;
        goto LABEL_14;
      }
    }
    v2 = v10;
    v8 = 0;
  }
LABEL_14:
  if ( v2 != v12 )
    _libc_free((unsigned __int64)v2);
  return v8;
}
