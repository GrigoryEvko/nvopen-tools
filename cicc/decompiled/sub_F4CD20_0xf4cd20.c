// Function: sub_F4CD20
// Address: 0xf4cd20
//
__int64 __fastcall sub_F4CD20(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        _BYTE *a7,
        size_t a8)
{
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r14
  __int64 i; // rbx
  __int64 v16; // rdi
  __int64 result; // rax
  __int64 v18; // [rsp+0h] [rbp-60h]
  __int64 v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+10h] [rbp-50h] BYREF
  __int64 v21; // [rsp+18h] [rbp-48h]
  __int64 v22; // [rsp+20h] [rbp-40h]
  unsigned int v23; // [rsp+28h] [rbp-38h]

  if ( a2 )
  {
    v20 = 0;
    v21 = 0;
    v22 = 0;
    v23 = 0;
    sub_F4C4C0(a1, a2, (__int64)&v20, a7, a8, (__int64)a5);
    v18 = a3 + 8 * a4;
    if ( a3 != v18 )
    {
      v19 = a3;
      do
      {
        v14 = *(_QWORD *)(*(_QWORD *)v19 + 56LL);
        for ( i = *(_QWORD *)v19 + 48LL; i != v14; v14 = *(_QWORD *)(v14 + 8) )
        {
          v16 = v14 - 24;
          if ( !v14 )
            v16 = 0;
          sub_F460A0(v16, &v20, a5, v11, v12, v13);
        }
        v19 += 8;
      }
      while ( v18 != v19 );
    }
    return sub_C7D6A0(v21, 16LL * v23, 8);
  }
  return result;
}
