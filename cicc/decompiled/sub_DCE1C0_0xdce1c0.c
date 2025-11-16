// Function: sub_DCE1C0
// Address: 0xdce1c0
//
unsigned __int64 __fastcall sub_DCE1C0(__int64 *a1, unsigned __int64 *a2, __int64 *a3)
{
  unsigned __int64 v3; // r12
  unsigned __int16 v4; // bx
  __int64 *v5; // rax
  __int64 v6; // rsi
  __int64 v7; // r11
  unsigned int v8; // r10d
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r15
  __int64 v12; // r13
  __int64 v13; // r14
  __int64 v14; // rdi
  __int64 v15; // rdx
  _QWORD *v16; // r12
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v22; // [rsp+8h] [rbp-78h] BYREF
  _QWORD v23[2]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD v24[12]; // [rsp+20h] [rbp-60h] BYREF

  v3 = *a2;
  v4 = *(_WORD *)(*a2 + 24);
  if ( (unsigned __int16)(v4 - 9) <= 3u && *(_QWORD *)(v3 + 40) == 2 )
  {
    v5 = *(__int64 **)(v3 + 32);
    v6 = *v5;
    if ( !*(_WORD *)(*v5 + 24) )
    {
      v7 = *(_QWORD *)(v6 + 32);
      v8 = *(_DWORD *)(v7 + 32);
      v9 = *(_QWORD *)(v7 + 24);
      v10 = 1LL << ((unsigned __int8)v8 - 1);
      if ( v8 > 0x40 )
      {
        if ( (*(_QWORD *)(v9 + 8LL * ((v8 - 1) >> 6)) & v10) != 0 )
          return v3;
      }
      else if ( (v9 & v10) != 0 )
      {
        return v3;
      }
      v11 = v5[1];
      v12 = *a1;
      v13 = *a3;
      if ( (unsigned __int16)(v4 - 11) > 1u )
      {
        v14 = *(_QWORD *)(v12 + 16);
        v16 = sub_DA2960(v14, v6, *a3);
      }
      else
      {
        v14 = *(_QWORD *)(v12 + 8);
        v16 = sub_DA2700(v14, v6, *a3);
      }
      v17 = *(_QWORD *)(v12 + 24);
      v23[0] = v13;
      v22 = v11;
      if ( !*(_QWORD *)(v17 + 16) )
        sub_4263D6(v14, v6, v15);
      v18 = (*(__int64 (__fastcall **)(__int64, __int64 *, _QWORD *))(v17 + 24))(v17, &v22, v23);
      v24[1] = v16;
      v24[0] = v18;
      v23[0] = v24;
      v23[1] = 0x600000002LL;
      v3 = sub_DCD310(*(__int64 **)(v12 + 32), v4, (__int64)v23, v19, v20);
      if ( (_QWORD *)v23[0] != v24 )
        _libc_free(v23[0], v4);
    }
  }
  return v3;
}
