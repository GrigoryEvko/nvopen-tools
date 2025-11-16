// Function: sub_1F350E0
// Address: 0x1f350e0
//
__int64 __fastcall sub_1F350E0(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  __int64 v4; // rdi
  __int64 (*v5)(); // rax
  __int64 v7; // [rsp+0h] [rbp-D0h] BYREF
  __int64 v8; // [rsp+8h] [rbp-C8h] BYREF
  _BYTE *v9; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v10; // [rsp+18h] [rbp-B8h]
  _BYTE v11[176]; // [rsp+20h] [rbp-B0h] BYREF

  v3 = 0;
  if ( (unsigned int)((__int64)(*(_QWORD *)(a3 + 96) - *(_QWORD *)(a3 + 88)) >> 3) > 1 )
    return v3;
  v4 = *a1;
  v7 = 0;
  v9 = v11;
  v8 = 0;
  v10 = 0x400000000LL;
  v5 = *(__int64 (**)())(*(_QWORD *)v4 + 264LL);
  if ( v5 == sub_1D820E0 )
    return v3;
  if ( !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, _BYTE **, _QWORD))v5)(
          v4,
          a3,
          &v7,
          &v8,
          &v9,
          0) )
    LOBYTE(v3) = (_DWORD)v10 == 0;
  if ( v9 == v11 )
    return v3;
  _libc_free((unsigned __int64)v9);
  return v3;
}
