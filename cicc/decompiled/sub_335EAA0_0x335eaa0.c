// Function: sub_335EAA0
// Address: 0x335eaa0
//
unsigned __int64 __fastcall sub_335EAA0(unsigned __int64 *a1, __int64 a2)
{
  __int64 v2; // rsi
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  unsigned __int64 v6; // r12
  __int64 (*v7)(void); // rdx
  int v8; // eax
  __int64 v10; // [rsp+8h] [rbp-38h] BYREF
  int v11[9]; // [rsp+1Ch] [rbp-24h] BYREF

  v10 = a2;
  v2 = a1[7];
  v3 = (__int64)(v2 - a1[6]) >> 8;
  v11[0] = v3;
  if ( v2 == a1[8] )
  {
    sub_335E500(a1 + 6, v2, &v10, v11);
    v5 = a1[7];
    v2 = v5 - 256;
  }
  else
  {
    if ( v2 )
    {
      v4 = v10;
      *(_DWORD *)(v2 + 200) = v3;
      *(_QWORD *)(v2 + 8) = 0;
      *(_QWORD *)v2 = v4;
      *(_QWORD *)(v2 + 40) = v2 + 56;
      *(_QWORD *)(v2 + 16) = 0;
      *(_QWORD *)(v2 + 24) = 0;
      *(_QWORD *)(v2 + 32) = 0;
      *(_QWORD *)(v2 + 48) = 0x400000000LL;
      *(_QWORD *)(v2 + 120) = v2 + 136;
      *(_QWORD *)(v2 + 128) = 0x400000000LL;
      *(_QWORD *)(v2 + 204) = 0;
      *(_QWORD *)(v2 + 212) = 0;
      *(_QWORD *)(v2 + 220) = 0;
      *(_QWORD *)(v2 + 228) = 0;
      *(_QWORD *)(v2 + 236) = 0;
      *(_QWORD *)(v2 + 244) = 0;
      *(_WORD *)(v2 + 252) = 0;
      *(_BYTE *)(v2 + 254) = 4;
      v2 = a1[7];
    }
    v5 = v2 + 256;
    a1[7] = v2 + 256;
  }
  *(_QWORD *)(v5 - 248) = v2;
  v6 = a1[7];
  if ( !v10 || *(_DWORD *)(v10 + 24) == -11 )
  {
    *(_BYTE *)(v6 - 2) &= 0xFu;
    return v6 - 256;
  }
  else
  {
    v7 = *(__int64 (**)(void))(**(_QWORD **)(a1[74] + 16) + 544LL);
    LOBYTE(v8) = 0;
    if ( v7 != sub_2FE3110 )
      v8 = v7() & 0xF;
    *(_BYTE *)(v6 - 2) = (16 * v8) | *(_BYTE *)(v6 - 2) & 0xF;
    return v6 - 256;
  }
}
