// Function: sub_166FF00
// Address: 0x166ff00
//
__int64 __fastcall sub_166FF00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  int v6; // eax
  int v7; // edx
  unsigned int v8; // r8d
  unsigned int v9; // eax
  __int64 v10; // rcx
  int v12; // r8d
  __int64 v13; // rdx
  bool v14; // zf
  unsigned __int8 v15; // [rsp+Fh] [rbp-41h] BYREF
  _QWORD v16[2]; // [rsp+10h] [rbp-40h] BYREF
  __int64 (__fastcall *v17)(const __m128i **, const __m128i *, int); // [rsp+20h] [rbp-30h]
  __int64 (__fastcall *v18)(); // [rsp+28h] [rbp-28h]

  v5 = a2;
  v6 = *(_DWORD *)(a1 + 928);
  if ( v6 )
  {
    v7 = v6 - 1;
    a2 = *(_QWORD *)(a1 + 912);
    v8 = 1;
    v9 = (v6 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v10 = *(_QWORD *)(a2 + 8LL * v9);
    if ( v10 == a3 )
      return v8;
    v12 = 1;
    while ( v10 != -8 )
    {
      v9 = v7 & (v12 + v9);
      v10 = *(_QWORD *)(a2 + 8LL * v9);
      if ( a3 == v10 )
        return 1;
      ++v12;
    }
  }
  v8 = 1;
  if ( (*(_BYTE *)(a3 + 32) & 0xFu) - 7 <= 1 )
    return v8;
  if ( (!v5 || (*(_BYTE *)(v5 + 32) & 0xF) == 1 || sub_15E4F60(v5)) && !sub_15E4F60(a3) && !*(_BYTE *)(a1 + 961) )
  {
    v14 = *(_QWORD *)(a1 + 32) == 0;
    v15 = 0;
    v16[1] = &v15;
    v18 = sub_1671900;
    v16[0] = a1;
    v17 = sub_166FED0;
    if ( v14 )
      sub_4263D6(a3, a2, v13);
    (*(void (__fastcall **)(__int64, __int64, _QWORD *))(a1 + 40))(a1 + 16, a3, v16);
    if ( v17 )
      v17((const __m128i **)v16, (const __m128i *)v16, 3);
    return v15;
  }
  return 0;
}
