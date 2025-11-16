// Function: sub_97F930
// Address: 0x97f930
//
__int64 __fastcall sub_97F930(__int64 a1, _BYTE *a2, size_t a3, __int64 a4, char a5)
{
  _BYTE *v6; // rax
  size_t v7; // rdx
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // rbx
  size_t v11; // r12
  __int64 v12; // r15
  void *s2; // [rsp+8h] [rbp-48h]
  _BYTE *v16; // [rsp+10h] [rbp-40h] BYREF
  size_t n; // [rsp+18h] [rbp-38h]

  v6 = sub_97E150(a2, a3);
  n = v7;
  v16 = v6;
  if ( v7 )
  {
    v8 = sub_97F810((__int64 *)(a1 + 176), &v16, (unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD))sub_97E7A0);
    v9 = *(_QWORD *)(a1 + 184);
    v10 = v8;
    if ( v8 != v9 )
    {
      v11 = n;
      s2 = v16;
      do
      {
        v12 = v10;
        if ( *(_QWORD *)(v10 + 8) != v11 || v11 && memcmp(*(const void **)v10, s2, v11) )
          break;
        if ( *(_DWORD *)a4 == *(_DWORD *)(v10 + 32)
          && *(_BYTE *)(v10 + 36) == *(_BYTE *)(a4 + 4)
          && a5 == *(_BYTE *)(v10 + 40) )
        {
          return v12;
        }
        v10 += 64;
      }
      while ( v9 != v10 );
    }
  }
  return 0;
}
