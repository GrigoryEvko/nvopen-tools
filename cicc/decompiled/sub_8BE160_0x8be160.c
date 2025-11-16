// Function: sub_8BE160
// Address: 0x8be160
//
__int64 __fastcall sub_8BE160(__int64 a1, int a2, __int64 *a3, unsigned int a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 v9; // rax
  char v10; // dl
  _QWORD v12[60]; // [rsp+10h] [rbp-410h] BYREF
  _BYTE v13[8]; // [rsp+1F0h] [rbp-230h] BYREF
  __int64 v14; // [rsp+1F8h] [rbp-228h]
  _BOOL4 v15; // [rsp+21Ch] [rbp-204h]
  _BOOL4 v16; // [rsp+220h] [rbp-200h]
  int v17; // [rsp+22Ch] [rbp-1F4h]
  unsigned int v18; // [rsp+24Ch] [rbp-1D4h]
  __int64 v19; // [rsp+27Ch] [rbp-1A4h]
  unsigned int v20; // [rsp+28Ch] [rbp-194h]
  __int64 v21; // [rsp+2A8h] [rbp-178h]
  __int64 v22; // [rsp+2D8h] [rbp-148h]
  _QWORD *v23; // [rsp+338h] [rbp-E8h]

  memset(v12, 0, 0x1D8u);
  v12[19] = v12;
  v12[3] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v12[22]) |= 1u;
  sub_891F00((__int64)v13, (__int64)v12);
  v23 = sub_854B90();
  v8 = *a3;
  v18 = a4;
  v19 = v8;
  v17 = a2;
  v20 = dword_4F06650[0];
  v9 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v15 = (*(_BYTE *)(v9 + 6) & 2) != 0;
  v10 = *(_BYTE *)(v9 + 6);
  v21 = a1;
  v22 = *(_QWORD *)(v9 + 184);
  v16 = (v10 & 8) != 0;
  sub_8BA620((unsigned __int64)v13, a4, a5);
  return v14;
}
