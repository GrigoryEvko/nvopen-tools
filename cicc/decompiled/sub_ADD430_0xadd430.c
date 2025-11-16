// Function: sub_ADD430
// Address: 0xadd430
//
__int64 __fastcall sub_ADD430(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // r15
  __int64 *v4; // rbx
  unsigned __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r12
  _BYTE *v8; // rsi
  __int64 v9; // r12
  _BYTE *v11; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v12; // [rsp+18h] [rbp-B8h]
  _BYTE v13[176]; // [rsp+20h] [rbp-B0h] BYREF

  v3 = &a2[a3];
  v11 = v13;
  v12 = 0x1000000000LL;
  if ( v3 == a2 )
  {
    v6 = 0;
    v8 = v13;
  }
  else
  {
    v4 = a2;
    v5 = 16;
    v6 = 0;
    while ( 1 )
    {
      v7 = *v4;
      if ( v6 + 1 > v5 )
      {
        sub_C8D5F0(&v11, v13, v6 + 1, 8);
        v6 = (unsigned int)v12;
      }
      ++v4;
      *(_QWORD *)&v11[8 * v6] = v7;
      v6 = (unsigned int)(v12 + 1);
      LODWORD(v12) = v12 + 1;
      if ( v3 == v4 )
        break;
      v5 = HIDWORD(v12);
    }
    v8 = v11;
  }
  v9 = sub_B9C770(*(_QWORD *)(a1 + 8), v8, v6, 0, 1);
  if ( v11 != v13 )
    _libc_free(v11, v8);
  return v9;
}
