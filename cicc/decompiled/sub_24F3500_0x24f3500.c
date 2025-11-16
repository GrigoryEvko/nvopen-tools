// Function: sub_24F3500
// Address: 0x24f3500
//
void __fastcall sub_24F3500(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // r12
  __int64 v9; // rax
  _QWORD **v10; // r15
  __int64 v11; // r12
  __int64 *v12; // rax
  _BYTE *v13; // [rsp+0h] [rbp-60h] BYREF
  __int64 v14; // [rsp+8h] [rbp-58h]
  _BYTE v15[80]; // [rsp+10h] [rbp-50h] BYREF

  v6 = *(_QWORD *)(a1 + 16);
  v13 = v15;
  v14 = 0x400000000LL;
  if ( v6 )
  {
    v7 = 0;
    do
    {
      while ( 1 )
      {
        v8 = *(_QWORD *)(v6 + 24);
        if ( *(_BYTE *)v8 == 85 )
        {
          v9 = *(_QWORD *)(v8 - 32);
          if ( v9 )
          {
            if ( !*(_BYTE *)v9
              && *(_QWORD *)(v9 + 24) == *(_QWORD *)(v8 + 80)
              && (*(_BYTE *)(v9 + 33) & 0x20) != 0
              && *(_DWORD *)(v9 + 36) == 28 )
            {
              break;
            }
          }
        }
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          goto LABEL_13;
      }
      if ( v7 + 1 > (unsigned __int64)HIDWORD(v14) )
      {
        sub_C8D5F0((__int64)&v13, v15, v7 + 1, 8u, v7 + 1, a6);
        v7 = (unsigned int)v14;
      }
      *(_QWORD *)&v13[8 * v7] = v8;
      v7 = (unsigned int)(v14 + 1);
      LODWORD(v14) = v14 + 1;
      v6 = *(_QWORD *)(v6 + 8);
    }
    while ( v6 );
LABEL_13:
    v10 = (_QWORD **)v13;
    if ( (_DWORD)v7 )
    {
      v11 = (unsigned int)v7;
      v12 = (__int64 *)sub_BD5C60(a1);
      sub_24F34B0(v12, v10, v11);
      if ( v13 != v15 )
        _libc_free((unsigned __int64)v13);
    }
    else if ( v13 != v15 )
    {
      _libc_free((unsigned __int64)v13);
    }
  }
}
