// Function: sub_A59EA0
// Address: 0xa59ea0
//
void __fastcall sub_A59EA0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  bool v4; // zf
  _BYTE *v5; // rsi
  _BYTE *v6; // r12
  _BYTE *v7; // rbx
  __int64 v8; // rax
  __int64 *v9; // r14
  __int64 *v10; // rbx
  __int64 v11; // rax
  _BYTE *v12; // rsi
  _BYTE *v13; // [rsp+0h] [rbp-70h] BYREF
  __int64 v14; // [rsp+8h] [rbp-68h]
  _BYTE v15[96]; // [rsp+10h] [rbp-60h] BYREF

  if ( *(_BYTE *)a2 == 85 )
  {
    v3 = *(_QWORD *)(a2 - 32);
    if ( v3 )
    {
      if ( !*(_BYTE *)v3 && *(_QWORD *)(a2 + 80) == *(_QWORD *)(v3 + 24) && (*(_BYTE *)(v3 + 33) & 0x20) != 0 )
      {
        v8 = 4LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
        if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        {
          v10 = *(__int64 **)(a2 - 8);
          v9 = &v10[v8];
        }
        else
        {
          v9 = (__int64 *)a2;
          v10 = (__int64 *)(a2 - v8 * 8);
        }
        for ( ; v9 != v10; v10 += 4 )
        {
          v11 = *v10;
          if ( *v10 )
          {
            if ( *(_BYTE *)v11 == 24 )
            {
              v12 = *(_BYTE **)(v11 + 24);
              if ( (unsigned __int8)(*v12 - 5) <= 0x1Fu )
                sub_A59AF0(a1, v12);
            }
          }
        }
      }
    }
  }
  v4 = *(_QWORD *)(a2 + 48) == 0;
  v13 = v15;
  v14 = 0x400000000LL;
  if ( !v4 || (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
  {
    v5 = &v13;
    sub_B9AA80(a2, &v13);
    v6 = v13;
    v7 = &v13[16 * (unsigned int)v14];
    if ( v7 != v13 )
    {
      do
      {
        v5 = (_BYTE *)*((_QWORD *)v6 + 1);
        v6 += 16;
        sub_A59AF0(a1, v5);
      }
      while ( v6 != v7 );
      v6 = v13;
    }
    if ( v6 != v15 )
      _libc_free(v6, v5);
  }
}
