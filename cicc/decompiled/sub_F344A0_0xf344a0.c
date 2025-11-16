// Function: sub_F344A0
// Address: 0xf344a0
//
__int64 __fastcall sub_F344A0(__int64 *a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  __int64 v5; // r12
  __int64 *v6; // rbx
  __int64 result; // rax
  __int64 *v8; // r12
  _QWORD *v9; // rdi
  _BYTE *v10; // [rsp+0h] [rbp-70h] BYREF
  __int64 v11; // [rsp+8h] [rbp-68h]
  _BYTE v12[96]; // [rsp+10h] [rbp-60h] BYREF

  v5 = a2;
  v6 = a1;
  v10 = v12;
  v11 = 0x400000000LL;
  if ( a3 )
  {
    sub_F34190(a1, a2, (__int64)&v10, a4);
    a2 = (__int64)v10;
    result = sub_FFB3D0(a3, v10, (unsigned int)v11);
  }
  else
  {
    result = (__int64)sub_F34190(a1, a2, 0, a4);
  }
  v8 = &a1[v5];
  if ( a1 != v8 )
  {
    do
    {
      while ( 1 )
      {
        a2 = *v6;
        if ( !a3 )
          break;
        ++v6;
        result = sub_FFBF00(a3, a2);
        if ( v8 == v6 )
          goto LABEL_8;
      }
      v9 = (_QWORD *)*v6++;
      result = sub_AA5450(v9);
    }
    while ( v8 != v6 );
  }
LABEL_8:
  if ( v10 != v12 )
    return _libc_free(v10, a2);
  return result;
}
