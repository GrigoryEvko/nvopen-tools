// Function: sub_2D35160
// Address: 0x2d35160
//
void __fastcall sub_2D35160(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r15d
  unsigned int v7; // r13d
  int v9; // eax
  __int64 v10; // rdx
  unsigned int *v11; // rsi
  __int64 v12; // rax
  unsigned int v13; // esi
  __int64 v14; // [rsp+0h] [rbp-90h] BYREF
  _BYTE *v15; // [rsp+8h] [rbp-88h]
  __int64 v16; // [rsp+10h] [rbp-80h]
  _BYTE v17[120]; // [rsp+18h] [rbp-78h] BYREF

  v6 = a4;
  v7 = a3;
  v9 = *(_DWORD *)(a1 + 192);
  if ( v9 )
  {
    v14 = a1;
    v15 = v17;
    v16 = 0x400000000LL;
    sub_2D2BC70((__int64)&v14, a2, a3, a4, a5, a6);
  }
  else
  {
    v10 = *(unsigned int *)(a1 + 196);
    if ( (_DWORD)v10 != 16 )
    {
      if ( (_DWORD)v10 )
      {
        v11 = (unsigned int *)(a1 + 4);
        do
        {
          if ( a2 < *v11 )
            break;
          ++v9;
          v11 += 2;
        }
        while ( (_DWORD)v10 != v9 );
      }
      LODWORD(v14) = v9;
      *(_DWORD *)(a1 + 196) = sub_2D28A50(a1, (unsigned int *)&v14, v10, a2, v7, a4);
      return;
    }
    v14 = a1;
    v16 = 0x400000000LL;
    v12 = 0;
    v15 = v17;
    while ( 1 )
    {
      v13 = v12;
      if ( a2 < *(_DWORD *)(a1 + 8 * v12 + 4) )
        break;
      if ( ++v12 == 16 )
      {
        v13 = 16;
        break;
      }
    }
    sub_2D29C80((__int64)&v14, v13, v10, a4, a5, a6);
  }
  sub_2D35090((__int64)&v14, a2, v7, v6);
  if ( v15 != v17 )
    _libc_free((unsigned __int64)v15);
}
