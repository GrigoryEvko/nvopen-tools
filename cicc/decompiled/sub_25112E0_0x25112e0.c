// Function: sub_25112E0
// Address: 0x25112e0
//
__int64 __fastcall sub_25112E0(_QWORD *a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, char a6)
{
  _QWORD *v10; // rax
  __int64 *v11; // rdi
  __int64 result; // rax
  __int64 v13; // rax
  __int64 v14; // r12
  void (__fastcall *v15)(__int64, _QWORD *, __int64); // rax
  void (__fastcall *v16)(_QWORD *, _QWORD *, __int64); // rax
  _QWORD v17[2]; // [rsp+0h] [rbp-90h] BYREF
  void (__fastcall *v18)(__int64, _QWORD *, __int64); // [rsp+10h] [rbp-80h]
  __int64 (__fastcall *v19)(__int64 **, __int64); // [rsp+18h] [rbp-78h]
  _QWORD v20[2]; // [rsp+20h] [rbp-70h] BYREF
  __int64 (__fastcall *v21)(_QWORD *, _QWORD *, int); // [rsp+30h] [rbp-60h]
  __int64 (__fastcall *v22)(__int64 **, __int64); // [rsp+38h] [rbp-58h]
  _QWORD v23[2]; // [rsp+40h] [rbp-50h] BYREF
  void (__fastcall *v24)(_QWORD *, _QWORD *, __int64); // [rsp+50h] [rbp-40h]
  __int64 (__fastcall *v25)(__int64 **, __int64); // [rsp+58h] [rbp-38h]

  a1[5] = a1 + 7;
  a1[6] = 0x600000000LL;
  a1[13] = a2 + 312;
  a1[24] = a1 + 26;
  v10 = a1 + 35;
  v11 = a1 + 43;
  *(v11 - 13) = a3;
  *(v11 - 43) = a5;
  *(v11 - 42) = 0;
  *(v11 - 41) = 0;
  *(v11 - 40) = 0;
  *((_DWORD *)v11 - 78) = 0;
  *(v11 - 29) = (__int64)a4;
  *(v11 - 28) = 0;
  *(v11 - 27) = 0;
  *(v11 - 26) = 0;
  *(v11 - 25) = 0;
  *((_DWORD *)v11 - 48) = 0;
  *(v11 - 23) = 0;
  *(v11 - 22) = 0;
  *(v11 - 21) = 0;
  *((_DWORD *)v11 - 40) = 0;
  *(v11 - 18) = 0;
  *(v11 - 17) = 0;
  *(v11 - 16) = 0;
  *(v11 - 15) = 0;
  *((_DWORD *)v11 - 28) = 0;
  *(v11 - 12) = 0;
  *(v11 - 11) = (__int64)v10;
  *(v11 - 10) = 8;
  *((_DWORD *)v11 - 18) = 0;
  *((_BYTE *)v11 - 68) = 1;
  a1[43] = a1 + 45;
  sub_2506C40(v11, *(_BYTE **)(a2 + 232), *(_QWORD *)(a2 + 232) + *(_QWORD *)(a2 + 240));
  a1[47] = *(_QWORD *)(a2 + 264);
  a1[48] = *(_QWORD *)(a2 + 272);
  result = *(_QWORD *)(a2 + 280);
  a1[49] = result;
  if ( a6 )
  {
    v17[0] = a3;
    v20[0] = a3;
    v19 = sub_2506F90;
    v23[0] = a3;
    v18 = (void (__fastcall *)(__int64, _QWORD *, __int64))sub_25061A0;
    v22 = sub_2507170;
    v21 = sub_25061D0;
    v25 = sub_2507080;
    v24 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))sub_2506200;
    v13 = sub_A777F0(0x108u, a4);
    v14 = v13;
    if ( v13 )
    {
      *(_BYTE *)(v13 + 2) = 1;
      *(_WORD *)v13 = 257;
      v15 = v18;
      *(_QWORD *)(v14 + 24) = 0;
      if ( v15 )
      {
        v15(v14 + 8, v17, 2);
        *(_QWORD *)(v14 + 32) = v19;
        *(_QWORD *)(v14 + 24) = v18;
      }
      *(_QWORD *)(v14 + 56) = 0;
      if ( v21 )
      {
        v21((_QWORD *)(v14 + 40), v20, 2);
        *(_QWORD *)(v14 + 64) = v22;
        *(_QWORD *)(v14 + 56) = v21;
      }
      *(_QWORD *)(v14 + 88) = 0;
      if ( v24 )
      {
        v24((_QWORD *)(v14 + 72), v23, 2);
        *(_QWORD *)(v14 + 96) = v25;
        *(_QWORD *)(v14 + 88) = v24;
      }
      *(_QWORD *)(v14 + 104) = 0;
      *(_QWORD *)(v14 + 112) = 0;
      *(_QWORD *)(v14 + 120) = 0;
      *(_DWORD *)(v14 + 128) = 0;
      *(_QWORD *)(v14 + 136) = 0;
      *(_QWORD *)(v14 + 144) = 0;
      *(_QWORD *)(v14 + 152) = 0;
      *(_DWORD *)(v14 + 160) = 0;
      *(_QWORD *)(v14 + 168) = 0;
      *(_QWORD *)(v14 + 176) = 0;
      *(_QWORD *)(v14 + 184) = 0;
      *(_DWORD *)(v14 + 192) = 0;
      sub_3106C40(v14 + 200, v14, 0);
    }
    v16 = v24;
    a1[15] = v14;
    if ( v16 )
      v16(v23, v23, 3);
    if ( v21 )
      v21(v20, v20, 3);
    result = (__int64)v18;
    if ( v18 )
      return ((__int64 (__fastcall *)(_QWORD *, _QWORD *, __int64))v18)(v17, v17, 3);
  }
  return result;
}
