// Function: sub_F4F720
// Address: 0xf4f720
//
__int64 __fastcall sub_F4F720(__int64 **a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v4; // rax
  unsigned __int8 v5; // dl
  __int64 v6; // rax
  _BYTE *v7; // rdi
  __int64 v8; // r13
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v12; // [rsp+18h] [rbp-38h]

  v4 = sub_B12000(a2 + 72);
  v5 = *(_BYTE *)(v4 - 16);
  if ( (v5 & 2) != 0 )
    v6 = *(_QWORD *)(v4 - 32);
  else
    v6 = v4 - 16 - 8LL * ((v5 >> 2) & 0xF);
  v7 = *(_BYTE **)(v6 + 24);
  if ( *v7 == 12 )
  {
    v12 = sub_AF2C80((__int64)v7);
    if ( BYTE4(v12) )
    {
      v8 = *a1[1];
      v9 = **a1;
      v10 = sub_B11F60(a2 + 80);
      return sub_B0E430(v10, v9, v8, (_DWORD)v12 == 0);
    }
  }
  return v2;
}
