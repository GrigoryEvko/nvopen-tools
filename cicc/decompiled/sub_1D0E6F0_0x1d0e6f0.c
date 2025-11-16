// Function: sub_1D0E6F0
// Address: 0x1d0e6f0
//
__int64 __fastcall sub_1D0E6F0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rsi
  unsigned __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 (*v7)(void); // rdx
  int v8; // eax
  __int64 v10; // [rsp+8h] [rbp-38h] BYREF
  int v11[9]; // [rsp+1Ch] [rbp-24h] BYREF

  v10 = a2;
  v2 = a1[7];
  v3 = 0xF0F0F0F0F0F0F0F1LL * ((v2 - a1[6]) >> 4);
  v11[0] = -252645135 * ((v2 - a1[6]) >> 4);
  if ( v2 == a1[8] )
  {
    sub_1D0E140(a1 + 6, (__int64 *)v2, &v10, v11);
    v5 = a1[7];
    v2 = v5 - 272;
  }
  else
  {
    if ( v2 )
    {
      v4 = v10;
      *(_DWORD *)(v2 + 192) = v3;
      *(_BYTE *)(v2 + 236) &= 0xFCu;
      *(_QWORD *)v2 = v4;
      *(_QWORD *)(v2 + 32) = v2 + 48;
      *(_QWORD *)(v2 + 8) = 0;
      *(_QWORD *)(v2 + 16) = 0;
      *(_QWORD *)(v2 + 24) = 0;
      *(_QWORD *)(v2 + 40) = 0x400000000LL;
      *(_QWORD *)(v2 + 112) = v2 + 128;
      *(_QWORD *)(v2 + 120) = 0x400000000LL;
      *(_QWORD *)(v2 + 196) = 0;
      *(_QWORD *)(v2 + 204) = 0;
      *(_QWORD *)(v2 + 212) = 0;
      *(_QWORD *)(v2 + 220) = 0;
      *(_WORD *)(v2 + 228) = 0;
      *(_DWORD *)(v2 + 232) = 0;
      *(_QWORD *)(v2 + 240) = 0;
      *(_QWORD *)(v2 + 248) = 0;
      *(_QWORD *)(v2 + 256) = 0;
      *(_QWORD *)(v2 + 264) = 0;
      v2 = a1[7];
    }
    v5 = v2 + 272;
    a1[7] = v2 + 272;
  }
  *(_QWORD *)(v5 - 256) = v2;
  v6 = a1[7];
  if ( !v10 || *(_WORD *)(v10 + 24) == 0xFFF6 )
  {
    *(_DWORD *)(v6 - 40) = 0;
    return v6 - 272;
  }
  else
  {
    v7 = *(__int64 (**)(void))(**(_QWORD **)(a1[78] + 16) + 280LL);
    v8 = 0;
    if ( v7 != sub_1D0B190 )
      v8 = v7();
    *(_DWORD *)(v6 - 40) = v8;
    return v6 - 272;
  }
}
