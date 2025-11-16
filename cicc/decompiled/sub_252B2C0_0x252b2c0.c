// Function: sub_252B2C0
// Address: 0x252b2c0
//
__int64 __fastcall sub_252B2C0(__int64 a1, unsigned __int8 **a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  unsigned __int8 **v5; // r14
  unsigned __int8 **i; // rbx
  unsigned __int64 v9; // rax
  __int64 v10; // rdi
  unsigned __int64 v11; // rax
  int v12; // edx
  int v13; // edx
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rdx
  _QWORD v17[2]; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int64 v18; // [rsp+10h] [rbp-40h]
  __int64 v19; // [rsp+18h] [rbp-38h]

  v4 = 0x8000000000041LL;
  v5 = &a2[a3];
  for ( i = a2; v5 != i; ++i )
  {
    v11 = (unsigned __int64)*i;
    if ( !*i )
      return 1;
    v17[0] = a1;
    v17[1] = a4;
    v12 = *(unsigned __int8 *)v11;
    if ( (_BYTE)v12 == 22 )
    {
      v19 = 0;
    }
    else
    {
      if ( (unsigned __int8)v12 > 0x1Cu )
      {
        v15 = (unsigned int)(v12 - 34);
        if ( (unsigned __int8)v15 <= 0x33u )
        {
          if ( _bittest64(&v4, v15) )
          {
            v19 = 0;
            v9 = v11 & 0xFFFFFFFFFFFFFFFCLL | 1;
            goto LABEL_5;
          }
        }
      }
      v18 = 0;
      v19 = 0;
      v13 = *(unsigned __int8 *)v11;
      if ( !(_BYTE)v13
        || (unsigned __int8)v13 > 0x1Cu
        && (v14 = (unsigned int)(v13 - 34), (unsigned __int8)v14 <= 0x33u)
        && _bittest64(&v4, v14) )
      {
        v9 = v11 & 0xFFFFFFFFFFFFFFFCLL | 2;
        goto LABEL_5;
      }
    }
    v9 = v11 & 0xFFFFFFFFFFFFFFFCLL;
LABEL_5:
    v18 = v9;
    nullsub_1518();
    v10 = sub_252AE70(a1, v18, v19, a4, 1, 0, 1);
    if ( !v10
      || !(*(unsigned __int8 (__fastcall **)(__int64, __int64 (__fastcall *)(__int64 *, unsigned __int8 *), _QWORD *, __int64))(*(_QWORD *)v10 + 112LL))(
            v10,
            sub_25284A0,
            v17,
            2) )
    {
      return 1;
    }
  }
  return 0;
}
