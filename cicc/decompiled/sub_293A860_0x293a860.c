// Function: sub_293A860
// Address: 0x293a860
//
void __fastcall sub_293A860(__int64 a1, __int64 a2)
{
  _BYTE **v2; // rbx
  __int64 v3; // r12
  _BYTE *v4; // r15
  __int64 v5; // rcx
  unsigned int *v6; // r14
  unsigned int *i; // r13
  unsigned __int64 v8; // rsi
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // rsi
  unsigned __int8 *v12; // rsi
  __int64 v14; // [rsp+28h] [rbp-98h]
  unsigned __int8 *v15; // [rsp+38h] [rbp-88h] BYREF
  unsigned int *v16; // [rsp+40h] [rbp-80h] BYREF
  __int64 v17; // [rsp+48h] [rbp-78h]
  _BYTE v18[112]; // [rsp+50h] [rbp-70h] BYREF

  v16 = (unsigned int *)v18;
  v17 = 0x400000000LL;
  sub_B9A9D0(a1, (__int64)&v16);
  v2 = *(_BYTE ***)a2;
  v3 = 33555946;
  v14 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v14 )
  {
    do
    {
      v4 = *v2;
      if ( **v2 > 0x1Cu )
      {
        v5 = 4LL * (unsigned int)v17;
        v6 = v16;
        for ( i = &v16[v5]; i != v6; v6 += 4 )
        {
          v8 = *v6;
          if ( (unsigned int)v8 <= 0x19 && _bittest64(&v3, v8) )
            sub_B99FD0((__int64)v4, v8, *((_QWORD *)v6 + 1));
        }
        sub_B45260(v4, a1, 1);
        v9 = *(_QWORD *)(a1 + 48);
        if ( v9 && !*((_QWORD *)v4 + 6) )
        {
          v15 = *(unsigned __int8 **)(a1 + 48);
          sub_B96E90((__int64)&v15, v9, 1);
          v10 = (__int64)(v4 + 48);
          if ( v4 + 48 == (_BYTE *)&v15 )
          {
            if ( v15 )
              sub_B91220(v10, (__int64)v15);
          }
          else
          {
            v11 = *((_QWORD *)v4 + 6);
            if ( v11 )
            {
              sub_B91220(v10, v11);
              v10 = (__int64)(v4 + 48);
            }
            v12 = v15;
            *((_QWORD *)v4 + 6) = v15;
            if ( v12 )
              sub_B976B0((__int64)&v15, v12, v10);
          }
        }
      }
      ++v2;
    }
    while ( (_BYTE **)v14 != v2 );
  }
  if ( v16 != (unsigned int *)v18 )
    _libc_free((unsigned __int64)v16);
}
