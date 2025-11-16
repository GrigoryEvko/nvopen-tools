// Function: sub_1BC2100
// Address: 0x1bc2100
//
void __fastcall sub_1BC2100(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // rdi
  _DWORD *v10; // [rsp+0h] [rbp-40h] BYREF
  __int64 v11; // [rsp+8h] [rbp-38h]
  _DWORD v12[12]; // [rsp+10h] [rbp-30h] BYREF

  v11 = 0x400000001LL;
  v6 = *(unsigned int *)(a1 + 24);
  v7 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  v10 = v12;
  v8 = v7 + 40 * v6;
  v12[0] = -2;
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
        v7 += 40;
        if ( v7 == v8 )
          goto LABEL_7;
      }
      v9 = v7;
      v7 += 40;
      sub_1BB9EE0(v9, (__int64)&v10, a3, a4, a5, a6);
    }
    while ( v7 != v8 );
LABEL_7:
    if ( v10 != v12 )
      _libc_free((unsigned __int64)v10);
  }
}
