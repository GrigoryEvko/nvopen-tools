// Function: sub_190D620
// Address: 0x190d620
//
void __fastcall sub_190D620(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // rdi
  __int64 v10; // [rsp+8h] [rbp-58h]
  _BYTE *v11; // [rsp+18h] [rbp-48h] BYREF
  __int64 v12; // [rsp+20h] [rbp-40h]
  _BYTE v13[56]; // [rsp+28h] [rbp-38h] BYREF

  v6 = *(unsigned int *)(a1 + 24);
  v7 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  v8 = v7 + (v6 << 6);
  v11 = v13;
  v12 = 0x400000000LL;
  if ( v8 != v7 )
  {
    do
    {
      while ( 1 )
      {
        if ( v7 )
        {
          *(_DWORD *)v7 = -1;
          *(_QWORD *)(v7 + 8) = v10;
          *(_DWORD *)(v7 + 32) = 0;
          *(_BYTE *)(v7 + 16) = 0;
          *(_QWORD *)(v7 + 24) = v7 + 40;
          *(_DWORD *)(v7 + 36) = 4;
          if ( (_DWORD)v12 )
            break;
        }
        v7 += 64;
        if ( v7 == v8 )
          goto LABEL_7;
      }
      v9 = v7 + 24;
      v7 += 64;
      sub_1909410(v9, (__int64)&v11, a3, a4, a5, a6);
    }
    while ( v7 != v8 );
LABEL_7:
    if ( v11 != v13 )
      _libc_free((unsigned __int64)v11);
  }
}
