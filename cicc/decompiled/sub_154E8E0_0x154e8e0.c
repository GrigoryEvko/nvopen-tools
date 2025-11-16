// Function: sub_154E8E0
// Address: 0x154e8e0
//
void __fastcall sub_154E8E0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 *v5; // rbx
  __int64 *v6; // r14
  __int64 v7; // rax
  _BYTE *v8; // rsi
  bool v9; // zf
  _BYTE *v10; // rbx
  _BYTE *v11; // r12
  __int64 v12; // rsi
  _BYTE *v13; // [rsp+0h] [rbp-70h] BYREF
  __int64 v14; // [rsp+8h] [rbp-68h]
  _BYTE v15[96]; // [rsp+10h] [rbp-60h] BYREF

  if ( *(_BYTE *)(a2 + 16) == 78 )
  {
    v3 = *(_QWORD *)(a2 - 24);
    if ( !*(_BYTE *)(v3 + 16) && (*(_BYTE *)(v3 + 33) & 0x20) != 0 )
    {
      v4 = 3LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      {
        v5 = *(__int64 **)(a2 - 8);
        v6 = &v5[v4];
      }
      else
      {
        v6 = (__int64 *)a2;
        v5 = (__int64 *)(a2 - v4 * 8);
      }
      while ( v6 != v5 )
      {
        v7 = *v5;
        if ( *v5 )
        {
          if ( *(_BYTE *)(v7 + 16) == 19 )
          {
            v8 = *(_BYTE **)(v7 + 24);
            if ( (unsigned __int8)(*v8 - 4) <= 0x1Eu )
              sub_154E670(a1, (__int64)v8);
          }
        }
        v5 += 3;
      }
    }
  }
  v9 = *(_QWORD *)(a2 + 48) == 0;
  v13 = v15;
  v14 = 0x400000000LL;
  if ( !v9 || *(__int16 *)(a2 + 18) < 0 )
  {
    sub_161F840(a2, &v13);
    v10 = v13;
    v11 = &v13[16 * (unsigned int)v14];
    if ( v13 != v11 )
    {
      do
      {
        v12 = *((_QWORD *)v10 + 1);
        v10 += 16;
        sub_154E670(a1, v12);
      }
      while ( v11 != v10 );
      v11 = v13;
    }
    if ( v11 != v15 )
      _libc_free((unsigned __int64)v11);
  }
}
