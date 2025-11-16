// Function: sub_107CAD0
// Address: 0x107cad0
//
__int64 __fastcall sub_107CAD0(__int64 a1, char **a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 result; // rax
  __int64 v9; // r15
  __int64 v10; // rdx
  char *v11; // [rsp+0h] [rbp-70h] BYREF
  __int64 v12; // [rsp+8h] [rbp-68h]
  char v13; // [rsp+10h] [rbp-60h] BYREF
  _BYTE *v14; // [rsp+18h] [rbp-58h] BYREF
  __int64 v15; // [rsp+20h] [rbp-50h]
  _BYTE v16[16]; // [rsp+28h] [rbp-48h] BYREF
  __int64 v17; // [rsp+38h] [rbp-38h]

  v12 = 0x100000000LL;
  v6 = *(_QWORD *)(a1 + 8);
  v17 = 0x100000000LL;
  v7 = *(unsigned int *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  result = 9 * v7;
  v11 = &v13;
  v9 = v6 + 8 * result;
  v14 = v16;
  v15 = 0x400000000LL;
  if ( v9 != v6 )
  {
    do
    {
      if ( v6 )
      {
        *(_DWORD *)(v6 + 8) = 0;
        *(_QWORD *)v6 = v6 + 16;
        *(_DWORD *)(v6 + 12) = 1;
        v10 = (unsigned int)v12;
        if ( (_DWORD)v12 )
        {
          a2 = &v11;
          sub_10774E0(v6, (__int64)&v11, (unsigned int)v12, a4, a5, a6);
        }
        *(_DWORD *)(v6 + 32) = 0;
        *(_QWORD *)(v6 + 24) = v6 + 40;
        *(_DWORD *)(v6 + 36) = 4;
        if ( (_DWORD)v15 )
        {
          a2 = &v14;
          sub_10774E0(v6 + 24, (__int64)&v14, v10, a4, a5, a6);
        }
        *(_DWORD *)(v6 + 56) = v17;
        result = HIDWORD(v17);
        *(_DWORD *)(v6 + 60) = HIDWORD(v17);
      }
      v6 += 72;
    }
    while ( v6 != v9 );
    if ( v14 != v16 )
      result = _libc_free(v14, a2);
    if ( v11 != &v13 )
      return _libc_free(v11, a2);
  }
  return result;
}
