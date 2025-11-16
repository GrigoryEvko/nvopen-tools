// Function: sub_2B312D0
// Address: 0x2b312d0
//
void __fastcall sub_2B312D0(unsigned int a1, __int64 a2, __int64 i, __int64 a4, __int64 j, __int64 a6)
{
  __int64 v6; // r11
  __int64 v8; // r15
  __int64 v9; // r10
  unsigned __int64 v10; // r15
  _DWORD *v11; // rax
  __int64 v12; // r11
  _DWORD *v13; // rsi
  int v14; // eax
  _DWORD *v15; // [rsp+10h] [rbp-70h] BYREF
  __int64 v16; // [rsp+18h] [rbp-68h]
  _DWORD v17[24]; // [rsp+20h] [rbp-60h] BYREF

  v6 = a1;
  v8 = *(unsigned int *)(a2 + 8);
  v15 = v17;
  v16 = 0xC00000000LL;
  v9 = v8;
  v10 = a1 * v8;
  if ( v10 )
  {
    v11 = v17;
    if ( v10 > 0xC )
    {
      sub_C8D5F0((__int64)&v15, v17, v10, 4u, j, a6);
      v6 = a1;
      v11 = &v15[(unsigned int)v16];
      for ( i = (__int64)&v15[v10]; (_DWORD *)i != v11; ++v11 )
      {
LABEL_4:
        if ( v11 )
          *v11 = 0;
      }
    }
    else
    {
      i = (__int64)&v17[v10];
      if ( (_DWORD *)i != v17 )
        goto LABEL_4;
    }
    LODWORD(v16) = v10;
    v9 = *(unsigned int *)(a2 + 8);
  }
  if ( (_DWORD)v9 )
  {
    v12 = 4 * v6;
    LODWORD(a6) = 0;
    for ( j = 0; j != v9; ++j )
    {
      a4 = 4LL * (unsigned int)j;
      v13 = &v15[(unsigned int)a6];
      i = 0;
      if ( v12 )
      {
        while ( 1 )
        {
          v14 = *(_DWORD *)(*(_QWORD *)a2 + 4LL * (unsigned int)j);
          if ( v14 != -1 )
            v14 = i + a1 * v14;
          v13[i] = v14;
          if ( i == (unsigned __int64)(v12 - 4) >> 2 )
            break;
          ++i;
        }
      }
      a6 = a1 + (unsigned int)a6;
    }
  }
  sub_2B310D0(a2, (__int64)&v15, i, a4, j, a6);
  if ( v15 != v17 )
    _libc_free((unsigned __int64)v15);
}
