// Function: sub_2169210
// Address: 0x2169210
//
__int64 __fastcall sub_2169210(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 ***v5; // rdi
  __int64 *v6; // r15
  __int64 ***v7; // r9
  __int64 **v8; // r12
  char v9; // dl
  __int64 *v10; // r13
  int v11; // r12d
  int i; // r14d
  __int64 v13; // rdx
  __int64 ***v15; // r8
  __int64 ***v16; // rax
  __int64 ***v17; // rsi
  unsigned int v18; // [rsp+4h] [rbp-9Ch]
  __int64 *v19; // [rsp+10h] [rbp-90h]
  unsigned int v20; // [rsp+18h] [rbp-88h]
  int v21; // [rsp+1Ch] [rbp-84h]
  __int64 v22; // [rsp+20h] [rbp-80h] BYREF
  __int64 ***v23; // [rsp+28h] [rbp-78h]
  __int64 ***v24; // [rsp+30h] [rbp-70h]
  __int64 v25; // [rsp+38h] [rbp-68h]
  int v26; // [rsp+40h] [rbp-60h]
  _BYTE v27[88]; // [rsp+48h] [rbp-58h] BYREF

  v5 = (__int64 ***)v27;
  v18 = a4;
  v22 = 0;
  v23 = (__int64 ***)v27;
  v24 = (__int64 ***)v27;
  v25 = 4;
  v26 = 0;
  v19 = &a2[a3];
  if ( a2 == v19 )
    return 0;
  v6 = a2;
  v7 = (__int64 ***)v27;
  v20 = 0;
  do
  {
    while ( 1 )
    {
      v8 = (__int64 **)*v6;
      if ( *(_BYTE *)(*v6 + 16) <= 0x10u )
        goto LABEL_3;
      if ( v5 == v7 )
      {
        v15 = &v5[HIDWORD(v25)];
        if ( v15 != v5 )
        {
          v16 = v5;
          v17 = 0;
          while ( v8 != *v16 )
          {
            if ( *v16 == (__int64 **)-2LL )
              v17 = v16;
            if ( v15 == ++v16 )
            {
              if ( !v17 )
                goto LABEL_27;
              *v17 = v8;
              --v26;
              ++v22;
              goto LABEL_7;
            }
          }
          goto LABEL_3;
        }
LABEL_27:
        if ( HIDWORD(v25) < (unsigned int)v25 )
          break;
      }
      sub_16CCBA0((__int64)&v22, *v6);
      v7 = v24;
      v5 = v23;
      if ( v9 )
        goto LABEL_7;
LABEL_3:
      if ( v19 == ++v6 )
        goto LABEL_16;
    }
    ++HIDWORD(v25);
    *v15 = v8;
    ++v22;
LABEL_7:
    v10 = *v8;
    if ( *((_BYTE *)*v8 + 8) != 16 )
      v10 = sub_16463B0(*v8, v18);
    v21 = v10[4];
    if ( v21 > 0 )
    {
      v11 = 0;
      for ( i = 0; i != v21; ++i )
      {
        v13 = (__int64)v10;
        if ( *((_BYTE *)v10 + 8) == 16 )
          v13 = *(_QWORD *)v10[2];
        v11 += sub_1F43D80(a1[2], *a1, v13, a4);
      }
      v20 += v11;
    }
    v7 = v24;
    v5 = v23;
    ++v6;
  }
  while ( v19 != v6 );
LABEL_16:
  if ( v5 != v7 )
    _libc_free((unsigned __int64)v7);
  return v20;
}
