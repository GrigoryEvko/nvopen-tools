// Function: sub_D474B0
// Address: 0xd474b0
//
__int64 __fastcall sub_D474B0(__int64 a1)
{
  _QWORD *v1; // rsi
  _BYTE *v2; // r15
  _BYTE *v3; // r12
  __int64 v4; // rbx
  __int64 v5; // rdx
  _QWORD *v6; // rax
  _QWORD *i; // rdx
  __int64 v8; // rdx
  unsigned int v9; // r12d
  _BYTE *v11; // [rsp+10h] [rbp-60h] BYREF
  __int64 v12; // [rsp+18h] [rbp-58h]
  _BYTE v13[80]; // [rsp+20h] [rbp-50h] BYREF

  v1 = &v11;
  v11 = v13;
  v12 = 0x400000000LL;
  sub_D474A0(a1, (__int64)&v11);
  v2 = v11;
  v3 = &v11[8 * (unsigned int)v12];
  if ( v3 == v11 )
  {
    v9 = 1;
  }
  else
  {
    do
    {
      v4 = *(_QWORD *)(*(_QWORD *)v2 + 16LL);
      if ( v4 )
      {
        while ( 1 )
        {
          v5 = *(_QWORD *)(v4 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v5 - 30) <= 0xAu )
            break;
          v4 = *(_QWORD *)(v4 + 8);
          if ( !v4 )
            goto LABEL_18;
        }
        v1 = *(_QWORD **)(v5 + 40);
        if ( *(_BYTE *)(a1 + 84) )
        {
LABEL_5:
          v6 = *(_QWORD **)(a1 + 64);
          for ( i = &v6[*(unsigned int *)(a1 + 76)]; i != v6; ++v6 )
          {
            if ( v1 == (_QWORD *)*v6 )
              goto LABEL_9;
          }
LABEL_13:
          v2 = v11;
          v9 = 0;
          goto LABEL_14;
        }
LABEL_12:
        if ( !sub_C8CA60(a1 + 56, (__int64)v1) )
          goto LABEL_13;
LABEL_9:
        while ( 1 )
        {
          v4 = *(_QWORD *)(v4 + 8);
          if ( !v4 )
            break;
          v8 = *(_QWORD *)(v4 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v8 - 30) <= 0xAu )
          {
            v1 = *(_QWORD **)(v8 + 40);
            if ( *(_BYTE *)(a1 + 84) )
              goto LABEL_5;
            goto LABEL_12;
          }
        }
      }
LABEL_18:
      v2 += 8;
    }
    while ( v3 != v2 );
    v2 = v11;
    v9 = 1;
  }
LABEL_14:
  if ( v2 != v13 )
    _libc_free(v2, v1);
  return v9;
}
