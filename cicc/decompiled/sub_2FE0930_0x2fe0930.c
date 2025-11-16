// Function: sub_2FE0930
// Address: 0x2fe0930
//
__int64 __fastcall sub_2FE0930(__int64 *a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r12d
  __int64 v7; // rax
  __int64 (*v8)(); // rax
  unsigned int v11; // eax
  _BYTE *v12; // rdi
  __int64 v13; // [rsp+8h] [rbp-58h] BYREF
  _QWORD *v14; // [rsp+10h] [rbp-50h] BYREF
  __int64 v15; // [rsp+18h] [rbp-48h]
  _BYTE v16[64]; // [rsp+20h] [rbp-40h] BYREF

  v6 = 0;
  v15 = 0x400000000LL;
  v7 = *a1;
  v14 = v16;
  v8 = *(__int64 (**)())(v7 + 816);
  v13 = 0;
  if ( v8 == sub_2EC09D0 )
    return v6;
  v11 = ((__int64 (__fastcall *)(__int64 *, __int64, _QWORD **, __int64, __int64, __int64 *, __int64))v8)(
          a1,
          a2,
          &v14,
          a4,
          a5,
          &v13,
          a6);
  v12 = v14;
  v6 = v11;
  if ( (_BYTE)v11 )
  {
    if ( (_DWORD)v15 == 1 )
      *a3 = *v14;
    else
      v6 = 0;
  }
  if ( v12 == v16 )
    return v6;
  _libc_free((unsigned __int64)v12);
  return v6;
}
