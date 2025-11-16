// Function: sub_2A04CA0
// Address: 0x2a04ca0
//
char __fastcall sub_2A04CA0(__int64 a1)
{
  char result; // al
  char v2; // r13
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 *v7; // rbx
  __int64 v8; // r12
  __int64 *v9; // r15
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 *v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  char v25; // [rsp+Fh] [rbp-61h]
  __int64 *v26; // [rsp+10h] [rbp-60h] BYREF
  __int64 v27; // [rsp+18h] [rbp-58h]
  _BYTE v28[80]; // [rsp+20h] [rbp-50h] BYREF

  result = sub_D4B3D0(a1);
  if ( result )
  {
    v2 = qword_5009A48;
    if ( (_BYTE)qword_5009A48 )
    {
      v27 = 0x400000000LL;
      v26 = (__int64 *)v28;
      sub_D4C2F0(a1, (__int64)&v26);
      v7 = v26;
      v8 = 8LL * (unsigned int)v27;
      v9 = &v26[(unsigned __int64)v8 / 8];
      v10 = v8 >> 3;
      v11 = v8 >> 5;
      if ( v11 )
      {
        v12 = &v26[4 * v11];
        while ( (unsigned __int8)sub_F347A0(*v7, (__int64)&v26, v3, v4, v5, v6) )
        {
          if ( !(unsigned __int8)sub_F347A0(v7[1], (__int64)&v26, v21, v22, v23, v24) )
          {
            result = v9 == v7 + 1;
            goto LABEL_12;
          }
          if ( !(unsigned __int8)sub_F347A0(v7[2], (__int64)&v26, v13, v14, v15, v16) )
          {
            result = v9 == v7 + 2;
            goto LABEL_12;
          }
          if ( !(unsigned __int8)sub_F347A0(v7[3], (__int64)&v26, v17, v18, v19, v20) )
          {
            result = v9 == v7 + 3;
            goto LABEL_12;
          }
          v7 += 4;
          if ( v7 == v12 )
          {
            v10 = v9 - v7;
            goto LABEL_15;
          }
        }
        goto LABEL_11;
      }
LABEL_15:
      if ( v10 != 2 )
      {
        if ( v10 != 3 )
        {
          if ( v10 != 1 )
          {
            result = v2;
            goto LABEL_12;
          }
LABEL_26:
          result = sub_F347A0(*v7, (__int64)&v26, v3, v4, v5, v6);
          if ( result )
          {
LABEL_12:
            if ( v26 != (__int64 *)v28 )
            {
              v25 = result;
              _libc_free((unsigned __int64)v26);
              return v25;
            }
            return result;
          }
LABEL_11:
          result = v9 == v7;
          goto LABEL_12;
        }
        if ( !(unsigned __int8)sub_F347A0(*v7, (__int64)&v26, v3, v4, v5, v6) )
          goto LABEL_11;
        ++v7;
      }
      if ( !(unsigned __int8)sub_F347A0(*v7, (__int64)&v26, v3, v4, v5, v6) )
        goto LABEL_11;
      ++v7;
      goto LABEL_26;
    }
  }
  return result;
}
