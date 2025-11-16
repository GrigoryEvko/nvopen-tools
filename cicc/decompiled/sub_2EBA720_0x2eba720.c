// Function: sub_2EBA720
// Address: 0x2eba720
//
__int64 __fastcall sub_2EBA720(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  bool v8; // zf
  __int64 v9; // r12
  _BYTE *v10; // r14
  unsigned __int64 v11; // r9
  unsigned __int64 v12; // rdi
  unsigned __int64 v14; // [rsp+18h] [rbp-D8h]
  char *v15[2]; // [rsp+20h] [rbp-D0h] BYREF
  _BYTE v16[32]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE *v17; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v18; // [rsp+58h] [rbp-98h]
  _BYTE v19[48]; // [rsp+60h] [rbp-90h] BYREF
  __int128 v20; // [rsp+90h] [rbp-60h]
  _OWORD v21[5]; // [rsp+A0h] [rbp-50h] BYREF

  v6 = a1 + 200;
  v8 = *(_BYTE *)(a1 + 352) == 0;
  v15[0] = v16;
  v17 = v19;
  v15[1] = (char *)0x400000000LL;
  v18 = 0x600000000LL;
  v20 = 0;
  memset(v21, 0, 24);
  if ( v8 )
  {
    *(_QWORD *)(a1 + 208) = 0x400000000LL;
    *(_QWORD *)(a1 + 200) = a1 + 216;
    *(_QWORD *)(a1 + 248) = a1 + 264;
    *(_QWORD *)(a1 + 256) = 0x600000000LL;
    *(_QWORD *)(a1 + 320) = 0;
    *(_QWORD *)(a1 + 328) = 0;
    *(_BYTE *)(a1 + 336) = 0;
    *(_QWORD *)(a1 + 340) = 0;
    sub_2E6DCE0((__int64 *)&v17);
    *((_QWORD *)&v20 + 1) = 0;
    *(_QWORD *)&v21[0] = 0;
    *(_BYTE *)(a1 + 352) = 1;
  }
  else
  {
    sub_2EB3190(a1 + 200, v15, 0x400000000LL, a4, a5, a6);
    sub_2EB4950(a1 + 248, (__int64)&v17);
    *(_QWORD *)(a1 + 320) = *((_QWORD *)&v20 + 1);
    *(_QWORD *)(a1 + 328) = *(_QWORD *)&v21[0];
    *(_BYTE *)(a1 + 336) = BYTE8(v21[0]);
    *(_QWORD *)(a1 + 340) = *(_QWORD *)((char *)v21 + 12);
    sub_2E6DCE0((__int64 *)&v17);
    *((_QWORD *)&v20 + 1) = 0;
    *(_QWORD *)&v21[0] = 0;
  }
  v9 = (__int64)v17;
  v10 = &v17[8 * (unsigned int)v18];
  if ( v17 != v10 )
  {
    do
    {
      v11 = *((_QWORD *)v10 - 1);
      v10 -= 8;
      if ( v11 )
      {
        v12 = *(_QWORD *)(v11 + 24);
        if ( v12 != v11 + 40 )
        {
          v14 = v11;
          _libc_free(v12);
          v11 = v14;
        }
        j_j___libc_free_0(v11);
      }
    }
    while ( (_BYTE *)v9 != v10 );
    v10 = v17;
  }
  if ( v10 != v19 )
    _libc_free((unsigned __int64)v10);
  if ( v15[0] != v16 )
    _libc_free((unsigned __int64)v15[0]);
  *(_QWORD *)(a1 + 328) = a2;
  *(_DWORD *)(a1 + 344) = *(_DWORD *)(a2 + 120);
  sub_2EBA550(v6);
  return 0;
}
