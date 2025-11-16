// Function: sub_372FDD0
// Address: 0x372fdd0
//
void __fastcall sub_372FDD0(int *a1, unsigned __int64 a2)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rdx
  __int64 v5; // r8
  __int64 v6; // r9
  _BYTE *v7; // r14
  _BYTE *v8; // r12
  __int64 v9; // r15
  int *v10; // rax
  size_t v11; // rdx
  _BYTE *v12; // [rsp+0h] [rbp-50h] BYREF
  __int64 v13; // [rsp+8h] [rbp-48h]
  _BYTE v14[64]; // [rsp+10h] [rbp-40h] BYREF

  v12 = v14;
  v13 = 0x100000000LL;
  while ( sub_3214EE0(a2) )
  {
    v3 = (unsigned int)v13;
    v4 = (unsigned int)v13 + 1LL;
    if ( v4 > HIDWORD(v13) )
    {
      sub_C8D5F0((__int64)&v12, v14, v4, 8u, v5, v6);
      v3 = (unsigned int)v13;
    }
    *(_QWORD *)&v12[8 * v3] = a2;
    LODWORD(v13) = v13 + 1;
    a2 = sub_3214EE0(a2);
  }
  v7 = v12;
  v8 = &v12[8 * (unsigned int)v13];
  if ( v12 != v8 )
  {
    do
    {
      while ( 1 )
      {
        v9 = *((_QWORD *)v8 - 1);
        sub_372FCB0(a1, 0x43u);
        sub_372FCB0(a1, *(unsigned __int16 *)(v9 + 28));
        v10 = (int *)sub_372FC20(v9);
        if ( v11 )
          break;
        v8 -= 8;
        if ( v7 == v8 )
          goto LABEL_11;
      }
      v8 -= 8;
      sub_372FC80(a1, v10, v11);
    }
    while ( v7 != v8 );
LABEL_11:
    v8 = v12;
  }
  if ( v8 != v14 )
    _libc_free((unsigned __int64)v8);
}
