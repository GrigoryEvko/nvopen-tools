// Function: sub_2A36FC0
// Address: 0x2a36fc0
//
__int64 __fastcall sub_2A36FC0(__int64 a1)
{
  __int64 v1; // r12
  char *v2; // rax
  char v3; // al
  void *v4; // rdi
  __int64 v5; // r14
  unsigned __int64 v6; // rbx
  _BYTE *v7; // rax
  char v8; // al
  _BYTE *v9; // rsi
  unsigned __int8 v11; // [rsp+1Fh] [rbp-61h]
  __int64 v12; // [rsp+28h] [rbp-58h] BYREF
  void *src; // [rsp+30h] [rbp-50h] BYREF
  _BYTE *v14; // [rsp+38h] [rbp-48h]
  _BYTE *v15; // [rsp+40h] [rbp-40h]

  v1 = *(_QWORD *)(a1 + 80);
  src = 0;
  v14 = 0;
  if ( v1 )
    v1 -= 24;
  v15 = 0;
  v2 = (char *)sub_C94E20((__int64)qword_4F86470);
  if ( v2 )
    v3 = *v2;
  else
    v3 = qword_4F86470[2];
  if ( v3 )
    sub_2D20E60(a1);
  v11 = 0;
LABEL_8:
  v4 = src;
  if ( src != v14 )
    v14 = src;
  v5 = *(_QWORD *)(v1 + 56);
  v6 = *(_QWORD *)(v1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v5 != v6 )
  {
    while ( 1 )
    {
      if ( !v5 )
        BUG();
      if ( *(_BYTE *)(v5 - 24) != 60 )
        goto LABEL_12;
      v12 = v5 - 24;
      if ( !(unsigned __int8)sub_2A4D8A0(v5 - 24, 1) )
        goto LABEL_12;
      v7 = sub_C94E20((__int64)qword_4F86470);
      v8 = v7 ? *v7 : LOBYTE(qword_4F86470[2]);
      if ( v8 && !(unsigned __int8)sub_2A4D6D0(v12) )
        goto LABEL_12;
      v9 = v14;
      if ( v14 == v15 )
      {
        sub_2A36E30((__int64)&src, v14, &v12);
LABEL_12:
        v5 = *(_QWORD *)(v5 + 8);
        if ( v5 == v6 )
          goto LABEL_24;
      }
      else
      {
        if ( v14 )
        {
          *(_QWORD *)v14 = v12;
          v9 = v14;
        }
        v14 = v9 + 8;
        v5 = *(_QWORD *)(v5 + 8);
        if ( v5 == v6 )
        {
LABEL_24:
          v4 = src;
          if ( v14 != src )
          {
            sub_2A57B70(src);
            v11 = 1;
            goto LABEL_8;
          }
          break;
        }
      }
    }
  }
  if ( v4 )
    j_j___libc_free_0((unsigned __int64)v4);
  return v11;
}
