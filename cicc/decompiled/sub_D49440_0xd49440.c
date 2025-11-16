// Function: sub_D49440
// Address: 0xd49440
//
void __fastcall sub_D49440(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rsi
  _BYTE *v8; // r15
  _BYTE *v9; // r12
  unsigned __int64 v10; // rdi
  int v11; // eax
  __int64 v12; // rdi
  _BYTE *v13; // [rsp+0h] [rbp-60h] BYREF
  __int64 v14; // [rsp+8h] [rbp-58h]
  _BYTE v15[80]; // [rsp+10h] [rbp-50h] BYREF

  v7 = (__int64)&v13;
  v14 = 0x400000000LL;
  v13 = v15;
  sub_D47A20(a1, (__int64)&v13, a3, a4, a5, a6);
  v8 = v13;
  v9 = &v13[8 * (unsigned int)v14];
  if ( v9 != v13 )
  {
    do
    {
      v10 = *(_QWORD *)(*(_QWORD *)v8 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v10 == *(_QWORD *)v8 + 48LL )
      {
        v12 = 0;
      }
      else
      {
        if ( !v10 )
          BUG();
        v11 = *(unsigned __int8 *)(v10 - 24);
        v12 = v10 - 24;
        if ( (unsigned int)(v11 - 30) >= 0xB )
          v12 = 0;
      }
      v7 = 18;
      v8 += 8;
      sub_B99FD0(v12, 0x12u, a2);
    }
    while ( v9 != v8 );
    v8 = v13;
  }
  if ( v8 != v15 )
    _libc_free(v8, v7);
}
