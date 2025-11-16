// Function: sub_D00920
// Address: 0xd00920
//
__int64 __fastcall sub_D00920(__int64 a1, _BYTE *a2, __int64 a3)
{
  __int64 v3; // r14
  unsigned __int64 v4; // rax
  __int64 v5; // r12
  int v6; // eax
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 *v9; // r13
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // rdx
  int v12; // eax
  unsigned int v13; // r15d
  unsigned int v14; // r12d
  _BYTE *v17; // [rsp+8h] [rbp-88h]
  int v18; // [rsp+10h] [rbp-80h]
  _BYTE *v19; // [rsp+20h] [rbp-70h] BYREF
  __int64 v20; // [rsp+28h] [rbp-68h]
  _BYTE v21[96]; // [rsp+30h] [rbp-60h] BYREF

  v3 = *(_QWORD *)(a1 + 40);
  v17 = a2;
  v4 = *(_QWORD *)(v3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v4 == v3 + 48 )
  {
    v14 = 1;
    v19 = v21;
  }
  else
  {
    if ( !v4 )
      BUG();
    v5 = v4 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v4 - 24) - 30 > 0xA )
    {
      HIDWORD(v20) = 6;
      v19 = v21;
      v12 = 0;
      v18 = 0;
    }
    else
    {
      v6 = sub_B46E30(v5);
      a2 = v21;
      v9 = (__int64 *)v21;
      v20 = 0x600000000LL;
      v10 = v6;
      v11 = v6;
      v12 = 0;
      v18 = v10;
      v19 = v21;
      if ( v10 > 6 )
      {
        sub_C8D5F0((__int64)&v19, v21, v11, 8u, v7, v8);
        v12 = v20;
        v9 = (__int64 *)&v19[8 * (unsigned int)v20];
      }
      if ( (_DWORD)v10 )
      {
        v13 = 0;
        do
        {
          if ( v9 )
          {
            a2 = (_BYTE *)v13;
            *v9 = sub_B46EC0(v5, v13);
          }
          ++v13;
          ++v9;
        }
        while ( (_DWORD)v10 != v13 );
        v12 = v20;
      }
    }
    v14 = 1;
    LODWORD(v20) = v18 + v12;
    if ( v18 + v12 )
    {
      a2 = (_BYTE *)v3;
      v14 = sub_D0E9A0(&v19, v3, 0, v17, a3) ^ 1;
    }
  }
  if ( v19 != v21 )
    _libc_free(v19, a2);
  return v14;
}
