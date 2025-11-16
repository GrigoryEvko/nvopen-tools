// Function: sub_39C2FF0
// Address: 0x39c2ff0
//
void __fastcall sub_39C2FF0(int *a1, unsigned __int64 a2)
{
  __int64 v3; // rax
  int v4; // r8d
  int v5; // r9d
  _BYTE *v6; // r14
  _BYTE *v7; // r12
  __int64 v8; // r15
  int *v9; // rax
  size_t v10; // rdx
  _BYTE *v11; // [rsp+0h] [rbp-50h] BYREF
  __int64 v12; // [rsp+8h] [rbp-48h]
  _BYTE v13[64]; // [rsp+10h] [rbp-40h] BYREF

  v11 = v13;
  v12 = 0x100000000LL;
  while ( sub_3981CC0(a2) )
  {
    v3 = (unsigned int)v12;
    if ( (unsigned int)v12 >= HIDWORD(v12) )
    {
      sub_16CD150((__int64)&v11, v13, 0, 8, v4, v5);
      v3 = (unsigned int)v12;
    }
    *(_QWORD *)&v11[8 * v3] = a2;
    LODWORD(v12) = v12 + 1;
    a2 = sub_3981CC0(a2);
  }
  v6 = v11;
  v7 = &v11[8 * (unsigned int)v12];
  if ( v11 != v7 )
  {
    do
    {
      while ( 1 )
      {
        v8 = *((_QWORD *)v7 - 1);
        sub_39C2ED0(a1, 0x43u);
        sub_39C2ED0(a1, *(unsigned __int16 *)(v8 + 28));
        v9 = (int *)sub_39C2E60(v8);
        if ( v10 )
          break;
        v7 -= 8;
        if ( v6 == v7 )
          goto LABEL_11;
      }
      v7 -= 8;
      sub_39C2EA0(a1, v9, v10);
    }
    while ( v6 != v7 );
LABEL_11:
    v7 = v11;
  }
  if ( v7 != v13 )
    _libc_free((unsigned __int64)v7);
}
