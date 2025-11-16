// Function: sub_31B0790
// Address: 0x31b0790
//
__int64 __fastcall sub_31B0790(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rax
  _QWORD *v6; // rdx
  _QWORD *v7; // rsi
  _QWORD *v8; // rcx
  _QWORD *v9; // rax
  _QWORD *v10; // rax
  __int64 v11; // rax
  bool v12; // zf
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 result; // rax
  _QWORD v17[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 (__fastcall *v18)(_QWORD *, _QWORD *, int); // [rsp+10h] [rbp-30h]
  void *v19; // [rsp+18h] [rbp-28h]

  *(_QWORD *)(a1 + 48) = a3;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_BYTE *)(a1 + 64) = 0;
  *(_BYTE *)(a1 + 80) = 0;
  *(_BYTE *)(a1 + 96) = 0;
  *(_BYTE *)(a1 + 112) = 0;
  v5 = (_QWORD *)sub_22077B0(0x2A8u);
  v6 = v5;
  if ( v5 )
  {
    *v5 = a2;
    v7 = v5 + 66;
    v8 = v5 + 44;
    v5[1] = a2;
    v9 = v5 + 4;
    *(v9 - 2) = 0;
    *(v9 - 1) = 1;
    do
    {
      if ( v9 )
      {
        *v9 = -4;
        v9[1] = -3;
        v9[2] = -4;
        v9[3] = -3;
      }
      v9 += 5;
    }
    while ( v8 != v9 );
    v6[44] = v7;
    v6[47] = v6 + 49;
    v6[48] = 0x400000000LL;
    *((_WORD *)v6 + 260) = 256;
    v6[45] = 0;
    *((_BYTE *)v6 + 368) = 0;
    v6[67] = 0;
    v6[68] = 1;
    v6[66] = &unk_49DDBE8;
    v10 = v6 + 69;
    do
    {
      if ( v10 )
        *v10 = -4096;
      v10 += 2;
    }
    while ( v6 + 85 != v10 );
  }
  *(_QWORD *)(a1 + 120) = v6;
  v17[0] = a1;
  v19 = sub_31AF7F0;
  v18 = sub_31AF840;
  v11 = sub_318AEA0(a3, (__int64)v17);
  v12 = *(_BYTE *)(a1 + 64) == 0;
  *(_QWORD *)(a1 + 56) = v11;
  if ( v12 )
    *(_BYTE *)(a1 + 64) = 1;
  if ( v18 )
    v18(v17, v17, 3);
  v17[0] = a1;
  v19 = sub_31AF800;
  v18 = sub_31AF870;
  v13 = sub_318ADE0(a3, (__int64)v17);
  v12 = *(_BYTE *)(a1 + 80) == 0;
  *(_QWORD *)(a1 + 72) = v13;
  if ( v12 )
    *(_BYTE *)(a1 + 80) = 1;
  if ( v18 )
    v18(v17, v17, 3);
  v17[0] = a1;
  v19 = sub_31AF810;
  v18 = sub_31AF8A0;
  v14 = sub_318B130(a3, (__int64)v17);
  v12 = *(_BYTE *)(a1 + 96) == 0;
  *(_QWORD *)(a1 + 88) = v14;
  if ( v12 )
    *(_BYTE *)(a1 + 96) = 1;
  if ( v18 )
    v18(v17, v17, 3);
  v17[0] = a1;
  v19 = sub_31AF820;
  v18 = sub_31AF8D0;
  v15 = sub_318B3C0(a3, (__int64)v17);
  v12 = *(_BYTE *)(a1 + 112) == 0;
  *(_QWORD *)(a1 + 104) = v15;
  if ( v12 )
    *(_BYTE *)(a1 + 112) = 1;
  result = (__int64)v18;
  if ( v18 )
    return v18(v17, v17, 3);
  return result;
}
