// Function: sub_295C520
// Address: 0x295c520
//
__int64 __fastcall sub_295C520(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  unsigned int v4; // eax
  _BYTE *v5; // rdi
  unsigned int v7; // esi
  unsigned int v8; // edx
  unsigned int v9; // [rsp+8h] [rbp-68h] BYREF
  unsigned int v10; // [rsp+Ch] [rbp-64h] BYREF
  _QWORD v11[2]; // [rsp+10h] [rbp-60h] BYREF
  _BYTE v12[80]; // [rsp+20h] [rbp-50h] BYREF

  v11[0] = v12;
  v11[1] = 0xC00000000LL;
  v4 = sub_BC8C10(a1, (__int64)v11);
  if ( (_BYTE)v4 )
  {
    sub_F02DB0(&v9, qword_5005A68 - 1, qword_5005A68);
    v5 = (_BYTE *)v11[0];
    v7 = *(_DWORD *)(v11[0] + 4LL * (*(_QWORD *)(a1 - 32) != a2));
    v8 = *(_DWORD *)v11[0] + *(_DWORD *)(v11[0] + 4LL);
    if ( v8 && v7 <= v8 )
    {
      sub_F02DB0(&v10, v7, v8);
      v5 = (_BYTE *)v11[0];
      LOBYTE(v2) = v9 <= v10;
    }
    else
    {
      v2 = 0;
    }
  }
  else
  {
    v5 = (_BYTE *)v11[0];
    v2 = v4;
  }
  if ( v5 != v12 )
    _libc_free((unsigned __int64)v5);
  return v2;
}
