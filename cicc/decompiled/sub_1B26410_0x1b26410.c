// Function: sub_1B26410
// Address: 0x1b26410
//
__int64 __fastcall sub_1B26410(__int64 a1)
{
  char *v1; // rax
  char v2; // al
  _QWORD *v3; // r14
  _BYTE *v4; // rdi
  _QWORD *v5; // r12
  __int64 v6; // r15
  unsigned __int64 v7; // rbx
  _BYTE *v8; // rax
  char v9; // al
  _BYTE *v10; // rsi
  unsigned __int8 v12; // [rsp+1Fh] [rbp-71h]
  __int64 v14; // [rsp+38h] [rbp-58h] BYREF
  void *src; // [rsp+40h] [rbp-50h] BYREF
  _BYTE *v16; // [rsp+48h] [rbp-48h]
  _BYTE *v17; // [rsp+50h] [rbp-40h]

  src = 0;
  v16 = 0;
  v17 = 0;
  v1 = (char *)sub_16D40F0((__int64)qword_4FBB510);
  if ( v1 )
    v2 = *v1;
  else
    v2 = qword_4FBB510[2];
  if ( v2 )
    sub_1CED850(a1);
  v12 = 0;
  v3 = (_QWORD *)(a1 + 72);
  while ( 1 )
  {
    v4 = src;
    if ( v16 != src )
      v16 = src;
    v5 = *(_QWORD **)(a1 + 80);
    if ( v3 == v5 )
      break;
    do
    {
      if ( !v5 )
        BUG();
      v6 = v5[3];
      v7 = v5[2] & 0xFFFFFFFFFFFFFFF8LL;
      if ( v7 != v6 )
      {
        while ( 1 )
        {
          if ( !v6 )
            BUG();
          if ( *(_BYTE *)(v6 - 8) != 53 )
            goto LABEL_12;
          v14 = v6 - 24;
          if ( !(unsigned __int8)sub_1B33710(v6 - 24, 0) )
            goto LABEL_12;
          v8 = sub_16D40F0((__int64)qword_4FBB510);
          v9 = v8 ? *v8 : LOBYTE(qword_4FBB510[2]);
          if ( v9 && !(unsigned __int8)sub_1B33670(v14) )
            goto LABEL_12;
          v10 = v16;
          if ( v16 == v17 )
          {
            sub_186B0F0((__int64)&src, v16, &v14);
LABEL_12:
            v6 = *(_QWORD *)(v6 + 8);
            if ( v7 == v6 )
              break;
          }
          else
          {
            if ( v16 )
            {
              *(_QWORD *)v16 = v14;
              v10 = v16;
            }
            v16 = v10 + 8;
            v6 = *(_QWORD *)(v6 + 8);
            if ( v7 == v6 )
              break;
          }
        }
      }
      v5 = (_QWORD *)v5[1];
    }
    while ( v3 != v5 );
    v4 = src;
    if ( v16 == src )
      break;
    sub_1B3B3D0(src);
    v12 = 1;
  }
  if ( v4 )
    j_j___libc_free_0(v4, v17 - v4);
  return v12;
}
