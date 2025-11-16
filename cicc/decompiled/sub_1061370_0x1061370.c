// Function: sub_1061370
// Address: 0x1061370
//
__int64 __fastcall sub_1061370(__int64 a1, __int64 a2, __int64 a3)
{
  int v5; // eax
  __int64 v6; // rcx
  int v7; // edx
  unsigned int v8; // r8d
  unsigned int v9; // eax
  __int64 v10; // rdi
  int v12; // r8d
  __int64 v13; // rax
  __int64 v14; // rdi
  unsigned __int8 v15; // [rsp+Fh] [rbp-41h] BYREF
  _QWORD v16[2]; // [rsp+10h] [rbp-40h] BYREF
  __int64 (__fastcall *v17)(const __m128i **, const __m128i *, int); // [rsp+20h] [rbp-30h]
  __int64 (__fastcall *v18)(); // [rsp+28h] [rbp-28h]

  v5 = *(_DWORD *)(a1 + 912);
  v6 = *(_QWORD *)(a1 + 896);
  if ( v5 )
  {
    v7 = v5 - 1;
    v8 = 1;
    v9 = (v5 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v10 = *(_QWORD *)(v6 + 8LL * v9);
    if ( a3 == v10 )
      return v8;
    v12 = 1;
    while ( v10 != -4096 )
    {
      v9 = v7 & (v12 + v9);
      v10 = *(_QWORD *)(v6 + 8LL * v9);
      if ( a3 == v10 )
        return 1;
      ++v12;
    }
  }
  v8 = 1;
  if ( (*(_BYTE *)(a3 + 32) & 0xFu) - 7 > 1 )
  {
    if ( a2 && (*(_BYTE *)(a2 + 32) & 0xF) != 1 && !sub_B2FC80(a2) )
      return 0;
    if ( sub_B2FC80(a3) )
      return 0;
    v8 = *(unsigned __int8 *)(a1 + 1002);
    if ( (_BYTE)v8 )
    {
      return 0;
    }
    else
    {
      v13 = *(_QWORD *)(a1 + 40);
      v15 = 0;
      if ( (v13 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v16[0] = a1;
        v14 = a1 + 16;
        v16[1] = &v15;
        v18 = sub_1063790;
        v17 = sub_1061340;
        if ( (v13 & 2) == 0 )
          v14 = *(_QWORD *)(a1 + 16);
        (*(void (__fastcall **)(__int64, __int64, _QWORD *))(v13 & 0xFFFFFFFFFFFFFFF8LL))(v14, a3, v16);
        if ( v17 )
          v17((const __m128i **)v16, (const __m128i *)v16, 3);
        return v15;
      }
    }
  }
  return v8;
}
