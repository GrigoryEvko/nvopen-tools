// Function: sub_F70320
// Address: 0xf70320
//
__int64 __fastcall sub_F70320(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v6; // edi
  __int64 v7; // rax
  __int64 v8; // r13
  unsigned int v9; // r14d
  _QWORD v11[2]; // [rsp+0h] [rbp-60h] BYREF
  _BYTE v12[32]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v13; // [rsp+30h] [rbp-30h]

  v6 = *(_DWORD *)(a3 + 40);
  v7 = (unsigned int)(v6 - 1);
  if ( (unsigned int)v7 > 0xF )
    BUG();
  v8 = *(_QWORD *)(a2 + 8);
  v9 = dword_3F8AE00[v7];
  v11[0] = sub_F70230(v6, *(_QWORD *)(v8 + 24), *(unsigned int *)(a3 + 44), (__int64)dword_3F8AE00, a5);
  v11[1] = a2;
  v13 = 257;
  return ((__int64 (__fastcall *)(__int64, _QWORD, __int64, _QWORD *, __int64, _BYTE *))sub_1061220)(
           a1,
           v9,
           v8,
           v11,
           2,
           v12);
}
