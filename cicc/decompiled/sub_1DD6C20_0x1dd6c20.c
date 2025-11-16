// Function: sub_1DD6C20
// Address: 0x1dd6c20
//
__int64 __fastcall sub_1DD6C20(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 v3; // rax
  __int64 (*v5)(); // rax
  __int64 v6; // rdi
  __int64 (*v7)(); // rax
  __int64 v9; // [rsp+0h] [rbp-E0h] BYREF
  __int64 v10; // [rsp+8h] [rbp-D8h] BYREF
  _BYTE *v11; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v12; // [rsp+18h] [rbp-C8h]
  _BYTE v13[192]; // [rsp+20h] [rbp-C0h] BYREF

  v2 = *(unsigned __int8 *)(a2 + 180);
  if ( (_BYTE)v2 )
  {
    return 0;
  }
  else
  {
    v3 = *(_QWORD *)(a1 + 56);
    if ( (*(_BYTE *)(*(_QWORD *)(v3 + 8) + 640LL) & 1) == 0 )
    {
      v5 = *(__int64 (**)())(**(_QWORD **)(v3 + 16) + 40LL);
      if ( v5 == sub_1D00B00 )
      {
        v9 = 0;
        v11 = v13;
        v10 = 0;
        v12 = 0x400000000LL;
        BUG();
      }
      v9 = 0;
      v6 = v5();
      v11 = v13;
      v10 = 0;
      v12 = 0x400000000LL;
      v7 = *(__int64 (**)())(*(_QWORD *)v6 + 264LL);
      if ( v7 != sub_1D820E0 )
      {
        if ( !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, _BYTE **, _QWORD))v7)(
                v6,
                a1,
                &v9,
                &v10,
                &v11,
                0) )
        {
          v2 = 1;
          if ( v9 )
            LOBYTE(v2) = v10 != v9;
        }
        if ( v11 != v13 )
          _libc_free((unsigned __int64)v11);
      }
    }
  }
  return v2;
}
