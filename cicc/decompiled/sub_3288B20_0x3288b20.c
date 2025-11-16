// Function: sub_3288B20
// Address: 0x3288b20
//
__int64 __fastcall sub_3288B20(
        int a1,
        int a2,
        int a3,
        int a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        int a9)
{
  __int64 v14; // rax
  __int16 v15; // dx
  __int64 v16; // rax
  int v17; // esi
  bool v19; // al
  int v20; // [rsp+8h] [rbp-48h]
  __int16 v21; // [rsp+10h] [rbp-40h] BYREF
  __int64 v22; // [rsp+18h] [rbp-38h]

  v14 = *(_QWORD *)(a5 + 48) + 16LL * (unsigned int)a6;
  v15 = *(_WORD *)v14;
  v16 = *(_QWORD *)(v14 + 8);
  v21 = v15;
  v22 = v16;
  if ( v15 )
  {
    v17 = ((unsigned __int16)(v15 - 17) < 0xD4u) + 205;
  }
  else
  {
    v20 = a3;
    v19 = sub_30070B0((__int64)&v21);
    a3 = v20;
    v17 = 205 - (!v19 - 1);
  }
  return sub_340EC60(a1, v17, a2, a3, a4, a9, a5, a6, a7, a8);
}
