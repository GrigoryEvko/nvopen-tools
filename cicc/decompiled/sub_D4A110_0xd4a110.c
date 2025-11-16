// Function: sub_D4A110
// Address: 0xd4a110
//
__int64 __fastcall sub_D4A110(__int64 a1, const void *a2, size_t a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BYTE *v6; // rax
  unsigned __int8 v7; // dl
  int v8; // edx
  __int64 v10; // [rsp+10h] [rbp-10h]

  v6 = sub_D49780(a1, a2, a3, a4, a5, a6);
  if ( v6 )
  {
    v7 = *(v6 - 16);
    if ( (v7 & 2) != 0 )
    {
      v8 = *((_DWORD *)v6 - 6);
      if ( v8 == 1 )
        return 0;
      if ( v8 == 2 )
        return *((_QWORD *)v6 - 4) + 8LL;
    }
    else
    {
      if ( ((*((_WORD *)v6 - 8) >> 6) & 0xF) == 1 )
        return 0;
      if ( ((*((_WORD *)v6 - 8) >> 6) & 0xF) == 2 )
        return (__int64)&v6[-8 * ((v7 >> 2) & 0xF) - 8];
    }
    BUG();
  }
  return v10;
}
