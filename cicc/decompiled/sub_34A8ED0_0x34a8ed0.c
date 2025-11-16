// Function: sub_34A8ED0
// Address: 0x34a8ed0
//
void __fastcall sub_34A8ED0(__int64 a1, unsigned __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v7; // r14
  unsigned int v9; // esi
  __int64 v10; // rdx
  unsigned __int64 *v11; // rax
  unsigned __int64 *v12; // rax
  __int64 v13; // [rsp+0h] [rbp-90h] BYREF
  _BYTE *v14; // [rsp+8h] [rbp-88h]
  __int64 v15; // [rsp+10h] [rbp-80h]
  _BYTE v16[120]; // [rsp+18h] [rbp-78h] BYREF

  v7 = a4;
  v9 = *(_DWORD *)(a1 + 192);
  if ( v9 )
  {
    v13 = a1;
    v14 = v16;
    v15 = 0x400000000LL;
    sub_34A3C90((__int64)&v13, a2, a3, a4, a5, a6);
  }
  else
  {
    v10 = *(unsigned int *)(a1 + 196);
    if ( (_DWORD)v10 != 11 )
    {
      if ( (_DWORD)v10 )
      {
        v11 = (unsigned __int64 *)(a1 + 8);
        do
        {
          if ( a2 <= *v11 )
            break;
          ++v9;
          v11 += 2;
        }
        while ( (_DWORD)v10 != v9 );
      }
      LODWORD(v13) = v9;
      *(_DWORD *)(a1 + 196) = sub_34A32D0(a1, (unsigned int *)&v13, v10, a2, a3, a4);
      return;
    }
    v13 = a1;
    v15 = 0x400000000LL;
    v12 = (unsigned __int64 *)(a1 + 8);
    v14 = v16;
    do
    {
      if ( a2 <= *v12 )
        break;
      ++v9;
      v12 += 2;
    }
    while ( v9 != 11 );
    sub_34A26E0((__int64)&v13, v9, v10, a4, a5, a6);
  }
  sub_34A8E00((__int64)&v13, a2, a3, v7);
  if ( v14 != v16 )
    _libc_free((unsigned __int64)v14);
}
