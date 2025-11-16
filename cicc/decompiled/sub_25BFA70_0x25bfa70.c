// Function: sub_25BFA70
// Address: 0x25bfa70
//
__int64 __fastcall sub_25BFA70(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, char a5)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  int v10; // eax
  bool v11; // cc
  unsigned __int64 v12; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v13; // [rsp+8h] [rbp-58h]
  unsigned __int64 v14; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-48h]
  __int64 v16; // [rsp+20h] [rbp-40h] BYREF
  __int64 v17; // [rsp+28h] [rbp-38h]
  __int64 v18; // [rsp+30h] [rbp-30h]
  int v19; // [rsp+38h] [rbp-28h]

  v7 = sub_9208B0(a2, a3);
  v17 = v8;
  v16 = v7;
  if ( (_BYTE)v8 != 1 && a5 )
  {
    v12 = a4;
    v15 = 64;
    v13 = 64;
    v14 = a4 + ((unsigned __int64)(v7 + 7) >> 3);
    sub_AADC30((__int64)&v16, (__int64)&v12, (__int64 *)&v14);
    v10 = v17;
    v11 = v13 <= 0x40;
    *(_BYTE *)(a1 + 32) = 1;
    *(_DWORD *)(a1 + 8) = v10;
    *(_QWORD *)a1 = v16;
    *(_DWORD *)(a1 + 24) = v19;
    *(_QWORD *)(a1 + 16) = v18;
    if ( !v11 && v12 )
      j_j___libc_free_0_0(v12);
    if ( v15 > 0x40 && v14 )
      j_j___libc_free_0_0(v14);
  }
  else
  {
    *(_BYTE *)(a1 + 32) = 0;
  }
  return a1;
}
