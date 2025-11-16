// Function: sub_E6C1C0
// Address: 0xe6c1c0
//
__int64 __fastcall sub_E6C1C0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // rdx
  char v9; // al
  unsigned __int8 v10; // cl
  __int64 v12; // [rsp+0h] [rbp-30h] BYREF
  __int64 v13; // [rsp+8h] [rbp-28h]
  __int64 *v14; // [rsp+10h] [rbp-20h]
  __int64 v15; // [rsp+18h] [rbp-18h]
  __int16 v16; // [rsp+20h] [rbp-10h]

  v6 = *(_QWORD *)(a1 + 152);
  v7 = *(_QWORD *)(v6 + 88);
  v8 = *(_QWORD *)(v6 + 96);
  v9 = *((_BYTE *)a2 + 32);
  v10 = *(_BYTE *)(a1 + 1907) ^ 1;
  if ( !v9 )
  {
    v16 = 256;
    return sub_E6BFC0((_DWORD *)a1, (__int64)&v12, 1, v10);
  }
  if ( v9 == 1 )
  {
    v12 = v7;
    v13 = v8;
    v16 = 261;
    return sub_E6BFC0((_DWORD *)a1, (__int64)&v12, 1, v10);
  }
  if ( *((_BYTE *)a2 + 33) == 1 )
  {
    a6 = a2[1];
    a2 = (__int64 *)*a2;
  }
  else
  {
    v9 = 2;
  }
  v13 = v8;
  v14 = a2;
  v12 = v7;
  v15 = a6;
  LOBYTE(v16) = 5;
  HIBYTE(v16) = v9;
  return sub_E6BFC0((_DWORD *)a1, (__int64)&v12, 1, v10);
}
