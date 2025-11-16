// Function: sub_12104A0
// Address: 0x12104a0
//
__int64 __fastcall sub_12104A0(__int64 a1, __int64 *a2, __int64 a3, unsigned __int64 a4, unsigned __int64 *a5)
{
  __int64 *v6; // r12
  __int64 v8; // r9
  __int64 v10; // rbx
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // r13
  const void *v14; // rsi
  unsigned __int64 v15; // rax
  _BYTE *v16; // rdi
  __int64 v17; // [rsp+0h] [rbp-90h]
  _QWORD *v19; // [rsp+10h] [rbp-80h] BYREF
  __int64 v20; // [rsp+18h] [rbp-78h]
  _BYTE v21[112]; // [rsp+20h] [rbp-70h] BYREF

  v6 = a2;
  if ( *((_BYTE *)a2 + 8) == 13 )
  {
    *a5 = (unsigned __int64)a2;
    return 0;
  }
  else
  {
    v8 = a3;
    v10 = a3;
    *a5 = 0;
    v19 = v21;
    v20 = 0x800000000LL;
    if ( a4 > 8 )
    {
      a2 = (__int64 *)v21;
      sub_C8D5F0((__int64)&v19, v21, a4, 8u, (__int64)a5, a3);
      v8 = a3;
    }
    v11 = v8 + 24 * a4;
    v12 = (unsigned int)v20;
    if ( v8 != v11 )
    {
      do
      {
        v13 = *(_QWORD *)(*(_QWORD *)(v10 + 8) + 8LL);
        if ( v12 + 1 > (unsigned __int64)HIDWORD(v20) )
        {
          a2 = (__int64 *)v21;
          v17 = v11;
          sub_C8D5F0((__int64)&v19, v21, v12 + 1, 8u, v11, v8);
          v12 = (unsigned int)v20;
          v11 = v17;
        }
        v10 += 24;
        v19[v12] = v13;
        v12 = (unsigned int)(v20 + 1);
        LODWORD(v20) = v20 + 1;
      }
      while ( v10 != v11 );
    }
    if ( (unsigned __int8)sub_BCB3E0((__int64)v6) )
    {
      v14 = v19;
      v15 = sub_BCF480(v6, v19, (unsigned int)v20, 0);
      v16 = v19;
      *a5 = v15;
      if ( v16 != v21 )
        _libc_free(v16, v14);
      return 0;
    }
    else
    {
      if ( v19 != (_QWORD *)v21 )
        _libc_free(v19, a2);
      return 1;
    }
  }
}
