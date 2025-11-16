// Function: sub_2FD7360
// Address: 0x2fd7360
//
__int64 __fastcall sub_2FD7360(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  __int64 v4; // rdi
  __int64 (*v5)(); // rax
  __int64 v7; // [rsp+0h] [rbp-E0h] BYREF
  __int64 v8; // [rsp+8h] [rbp-D8h] BYREF
  _BYTE *v9; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v10; // [rsp+18h] [rbp-C8h]
  _BYTE v11[192]; // [rsp+20h] [rbp-C0h] BYREF

  v3 = 0;
  if ( *(_DWORD *)(a3 + 120) <= 1u )
  {
    v4 = *a1;
    v7 = 0;
    v9 = v11;
    v8 = 0;
    v10 = 0x400000000LL;
    v5 = *(__int64 (**)())(*(_QWORD *)v4 + 344LL);
    if ( v5 != sub_2DB1AE0 )
    {
      v3 = ((__int64 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, _BYTE **, _QWORD))v5)(
             v4,
             a3,
             &v7,
             &v8,
             &v9,
             0);
      if ( (_BYTE)v3 )
      {
        v3 = 0;
      }
      else if ( !(_DWORD)v10 )
      {
        v3 = *(unsigned __int8 *)(a2 + 262) ^ 1;
      }
      if ( v9 != v11 )
        _libc_free((unsigned __int64)v9);
    }
  }
  return v3;
}
