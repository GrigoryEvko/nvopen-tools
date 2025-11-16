// Function: sub_84D980
// Address: 0x84d980
//
void __fastcall sub_84D980(__int64 *a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r13
  _BYTE *v4; // rax
  _BYTE *v5; // rdx
  __int64 v6; // rcx
  __int64 *v7; // r9
  _BYTE *v8; // r15
  __int64 v9; // [rsp+8h] [rbp-38h]

  v2 = a1[1];
  if ( v2 < a2 )
  {
    v3 = a1[2];
    v9 = *a1;
    v4 = (_BYTE *)sub_823970(a2);
    v8 = v4;
    if ( v3 > 0 )
    {
      v5 = (_BYTE *)v9;
      v6 = (__int64)&v4[v3];
      do
      {
        if ( v4 )
          *v4 = *v5;
        ++v4;
        ++v5;
      }
      while ( (_BYTE *)v6 != v4 );
    }
    sub_823A00(v9, v2, (__int64)v5, v6, v9, v7);
    *a1 = (__int64)v8;
    a1[1] = a2;
  }
}
