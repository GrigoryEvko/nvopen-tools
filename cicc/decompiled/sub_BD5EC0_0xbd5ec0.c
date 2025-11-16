// Function: sub_BD5EC0
// Address: 0xbd5ec0
//
__int64 __fastcall sub_BD5EC0(__int64 a1, unsigned __int8 (__fastcall *a2)(__int64, __int64), __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rbx
  unsigned __int8 (__fastcall *v5)(__int64, __int64); // r12
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  __int64 *v9; // rdi
  __int64 *v10; // r12
  __int64 *v11; // rbx
  __int64 v12; // rdi
  __int64 *v13; // [rsp+0h] [rbp-80h] BYREF
  __int64 v14; // [rsp+8h] [rbp-78h]
  _BYTE v15[112]; // [rsp+10h] [rbp-70h] BYREF

  result = 0x800000000LL;
  v4 = *(_QWORD *)(a1 + 16);
  v13 = (__int64 *)v15;
  v14 = 0x800000000LL;
  if ( v4 )
  {
    v5 = a2;
    do
    {
      while ( 1 )
      {
        if ( sub_BD2BE0(*(_QWORD *)(v4 + 24)) )
        {
          a2 = (unsigned __int8 (__fastcall *)(__int64, __int64))v4;
          if ( v5(a3, v4) )
            break;
        }
        v4 = *(_QWORD *)(v4 + 8);
        if ( !v4 )
          goto LABEL_9;
      }
      v7 = (unsigned int)v14;
      v8 = (unsigned int)v14 + 1LL;
      if ( v8 > HIDWORD(v14) )
      {
        a2 = (unsigned __int8 (__fastcall *)(__int64, __int64))v15;
        sub_C8D5F0(&v13, v15, v8, 8);
        v7 = (unsigned int)v14;
      }
      v13[v7] = v4;
      LODWORD(v14) = v14 + 1;
      v4 = *(_QWORD *)(v4 + 8);
    }
    while ( v4 );
LABEL_9:
    v9 = v13;
    result = (unsigned int)v14;
    v10 = &v13[(unsigned int)v14];
    if ( v10 != v13 )
    {
      v11 = v13;
      do
      {
        v12 = *v11++;
        result = sub_BD5D50(v12);
      }
      while ( v10 != v11 );
      v9 = v13;
    }
    if ( v9 != (__int64 *)v15 )
      return _libc_free(v9, a2);
  }
  return result;
}
