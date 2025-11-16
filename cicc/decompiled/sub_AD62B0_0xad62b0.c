// Function: sub_AD62B0
// Address: 0xad62b0
//
__int64 __fastcall sub_AD62B0(__int64 a1)
{
  unsigned __int8 v1; // al
  unsigned int v2; // eax
  unsigned __int64 v3; // rdx
  __int64 v4; // r12
  __int64 v6; // rax
  __int64 v7; // r12
  unsigned __int8 *v8; // rsi
  int v9; // eax
  unsigned __int64 v10; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v11; // [rsp+8h] [rbp-38h]

  v1 = *(_BYTE *)(a1 + 8);
  if ( v1 == 12 )
  {
    v2 = *(_DWORD *)(a1 + 8) >> 8;
    v11 = v2;
    if ( v2 > 0x40 )
    {
      sub_C43690(&v10, -1, 1);
    }
    else
    {
      v3 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v2;
      if ( !v2 )
        v3 = 0;
      v10 = v3;
    }
    v4 = sub_ACCFD0(*(__int64 **)a1, (__int64)&v10);
    if ( v11 > 0x40 )
    {
      if ( v10 )
        j_j___libc_free_0_0(v10);
    }
    return v4;
  }
  else if ( v1 <= 3u || v1 == 5 || (v1 & 0xFD) == 4 )
  {
    v6 = sub_BCAC60(a1);
    sub_C418D0(&v10, v6);
    v7 = sub_AC8EA0(*(__int64 **)a1, (__int64 *)&v10);
    sub_91D830(&v10);
    return v7;
  }
  else
  {
    v8 = (unsigned __int8 *)sub_AD62B0(*(_QWORD *)(a1 + 24));
    v9 = *(_DWORD *)(a1 + 32);
    BYTE4(v10) = *(_BYTE *)(a1 + 8) == 18;
    LODWORD(v10) = v9;
    return sub_AD5E10(v10, v8);
  }
}
