// Function: sub_3587420
// Address: 0x3587420
//
__int64 __fastcall sub_3587420(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v5; // rax
  float v6; // xmm0_4
  float v7; // xmm0_4
  unsigned __int64 v8; // r8
  unsigned __int64 v9; // rax
  __int64 (__fastcall **v10)(); // rax
  int v12; // eax
  __int64 **v13; // r13
  __int64 v14; // rax
  unsigned __int64 *v15; // rbx
  unsigned __int64 *v16; // r13
  unsigned __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // [rsp+8h] [rbp-258h]
  unsigned __int64 v21; // [rsp+18h] [rbp-248h] BYREF
  unsigned int v22[5]; // [rsp+20h] [rbp-240h] BYREF
  char v23; // [rsp+34h] [rbp-22Ch]
  _QWORD v24[2]; // [rsp+40h] [rbp-220h] BYREF
  char v25; // [rsp+50h] [rbp-210h]
  __int64 v26[4]; // [rsp+60h] [rbp-200h] BYREF
  _QWORD v27[10]; // [rsp+80h] [rbp-1E0h] BYREF
  unsigned __int64 *v28; // [rsp+D0h] [rbp-190h]
  unsigned int v29; // [rsp+D8h] [rbp-188h]
  char v30; // [rsp+E0h] [rbp-180h] BYREF

  sub_35847E0((__int64)v22, a3);
  if ( v23 && (v5 = (*(__int64 (__fastcall **)(_QWORD *, __int64))(*a2 + 16LL))(a2, a3)) != 0 )
  {
    v20 = v5;
    sub_35845F0((__int64)v24, v5, v22[0], v22[3]);
    if ( (v25 & 1) != 0 )
    {
      v12 = v24[0];
      *(_BYTE *)(a1 + 16) |= 1u;
      *(_DWORD *)a1 = v12;
      *(_QWORD *)(a1 + 8) = v24[1];
    }
    else
    {
      if ( v24[0] < 0LL )
        v6 = (float)(int)(v24[0] & 1 | (v24[0] >> 1)) + (float)(int)(v24[0] & 1 | (v24[0] >> 1));
      else
        v6 = (float)SLODWORD(v24[0]);
      v7 = v6 * *(float *)&v22[4];
      if ( v7 >= 9.223372e18 )
        v8 = (unsigned int)(int)(float)(v7 - 9.223372e18) ^ 0x8000000000000000LL;
      else
        v8 = (unsigned int)(int)v7;
      v21 = v8;
      if ( sub_2A61A10((__int64)(a2 + 136), v20, v22[0], 0, v8) )
      {
        v13 = (__int64 **)a2[161];
        v26[0] = a3;
        v26[1] = (__int64)&v21;
        v26[2] = (__int64)v22;
        v26[3] = (__int64)v24;
        v14 = sub_B2BE50(**v13);
        if ( sub_B6EA50(v14)
          || (v18 = sub_B2BE50(**v13),
              v19 = sub_B6F970(v18),
              (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v19 + 48LL))(v19)) )
        {
          sub_3586F00((__int64)v27, v26);
          sub_2EAFC50(v13, (__int64)v27);
          v15 = v28;
          v27[0] = &unk_49D9D40;
          v16 = &v28[10 * v29];
          if ( v28 != v16 )
          {
            do
            {
              v16 -= 10;
              v17 = v16[4];
              if ( (unsigned __int64 *)v17 != v16 + 6 )
                j_j___libc_free_0(v17);
              if ( (unsigned __int64 *)*v16 != v16 + 2 )
                j_j___libc_free_0(*v16);
            }
            while ( v15 != v16 );
            v16 = v28;
          }
          if ( v16 != (unsigned __int64 *)&v30 )
            _libc_free((unsigned __int64)v16);
        }
      }
      v9 = v21;
      *(_BYTE *)(a1 + 16) &= ~1u;
      *(_QWORD *)a1 = v9;
    }
  }
  else
  {
    v10 = sub_2241E40();
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = v10;
  }
  return a1;
}
