// Function: sub_1995230
// Address: 0x1995230
//
void __fastcall sub_1995230(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // rdi
  _QWORD *v10; // [rsp+0h] [rbp-50h] BYREF
  __int64 v11; // [rsp+8h] [rbp-48h]
  _QWORD v12[8]; // [rsp+10h] [rbp-40h] BYREF

  v6 = *(unsigned int *)(a1 + 24);
  v7 = *(_QWORD *)(a1 + 8);
  v11 = 0x400000001LL;
  *(_QWORD *)(a1 + 16) = 0;
  v10 = v12;
  v12[0] = -1;
  v8 = v7 + 56 * v6;
  if ( v8 != v7 )
  {
    do
    {
      while ( 1 )
      {
        if ( v7 )
        {
          *(_DWORD *)(v7 + 8) = 0;
          *(_QWORD *)v7 = v7 + 16;
          *(_DWORD *)(v7 + 12) = 4;
          if ( (_DWORD)v11 )
            break;
        }
        v7 += 56;
        if ( v7 == v8 )
          goto LABEL_7;
      }
      v9 = v7;
      v7 += 56;
      sub_19930D0(v9, (__int64)&v10, v6, a4, a5, a6);
    }
    while ( v7 != v8 );
LABEL_7:
    if ( v10 != v12 )
      _libc_free((unsigned __int64)v10);
  }
}
