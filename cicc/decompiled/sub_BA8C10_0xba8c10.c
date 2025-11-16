// Function: sub_BA8C10
// Address: 0xba8c10
//
__int64 __fastcall sub_BA8C10(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v9; // r15d
  __int64 v10; // rax
  __int64 v11; // rbx
  _QWORD v12[4]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v13; // [rsp+20h] [rbp-40h]

  if ( !sub_BA8B30(a1, a2, a3) )
  {
    v12[0] = a2;
    v9 = *(_DWORD *)(a1 + 320);
    v12[1] = a3;
    v13 = 261;
    v10 = sub_BD2DA0(136);
    v11 = v10;
    if ( v10 )
      sub_B2C3B0(v10, a4, 0, v9, (__int64)v12, a1);
    if ( (*(_BYTE *)(v11 + 33) & 0x20) == 0 )
      *(_QWORD *)(v11 + 120) = a5;
  }
  return a4;
}
