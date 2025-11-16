// Function: sub_26C63A0
// Address: 0x26c63a0
//
__int64 __fastcall sub_26C63A0(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // r15
  _QWORD *v8; // rdi
  __int64 v9; // rax
  unsigned int *v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rax
  float v13; // xmm0_4
  float v14; // xmm0_4
  unsigned __int64 v15; // r8
  unsigned __int64 v16; // rax
  __int64 (__fastcall **v17)(); // rax
  __int64 *v19; // r13
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned __int64 v24; // [rsp+18h] [rbp-248h] BYREF
  _DWORD v25[5]; // [rsp+20h] [rbp-240h] BYREF
  char v26; // [rsp+34h] [rbp-22Ch]
  __int64 v27; // [rsp+40h] [rbp-220h] BYREF
  char v28; // [rsp+50h] [rbp-210h]
  _QWORD v29[4]; // [rsp+60h] [rbp-200h] BYREF
  _QWORD v30[10]; // [rsp+80h] [rbp-1E0h] BYREF
  char v31[400]; // [rsp+D0h] [rbp-190h] BYREF

  sub_3143F80(v25, a3, a3);
  if ( v26
    && (v6 = (*(__int64 (__fastcall **)(_QWORD *, __int64))(*a2 + 16LL))(a2, a3), (v7 = v6) != 0)
    && ((v30[0] = __PAIR64__(v25[3], v25[0]), (v8 = *(_QWORD **)(v6 + 168)) == 0)
     || (v9 = sub_C1BA30(v8, (__int64)v30)) == 0
      ? (v10 = (unsigned int *)v30)
      : (v10 = (unsigned int *)(v9 + 16)),
        v11 = sub_26C2A80(v7 + 72, v10),
        v11 != v7 + 80) )
  {
    v12 = *(_QWORD *)(v11 + 40);
    v28 &= ~1u;
    v27 = v12;
    if ( v12 < 0 )
      v13 = (float)(v12 & 1 | (unsigned int)((unsigned __int64)v12 >> 1))
          + (float)(v12 & 1 | (unsigned int)((unsigned __int64)v12 >> 1));
    else
      v13 = (float)(int)v12;
    v14 = v13 * *(float *)&v25[4];
    if ( v14 >= 9.223372e18 )
      v15 = (unsigned int)(int)(float)(v14 - 9.223372e18) ^ 0x8000000000000000LL;
    else
      v15 = (unsigned int)(int)v14;
    v24 = v15;
    if ( (unsigned __int8)sub_2A61A10(a2 + 136, v7, v25[0], 0) )
    {
      v19 = (__int64 *)a2[161];
      v29[0] = a3;
      v29[1] = &v24;
      v29[2] = v25;
      v29[3] = &v27;
      v20 = *v19;
      v21 = sub_B2BE50(*v19);
      if ( sub_B6EA50(v21)
        || (v22 = sub_B2BE50(v20),
            v23 = sub_B6F970(v22),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v23 + 48LL))(v23)) )
      {
        sub_26C5F30((__int64)v30, (__int64)v29);
        sub_1049740(v19, (__int64)v30);
        v30[0] = &unk_49D9D40;
        sub_23FD590((__int64)v31);
      }
    }
    v16 = v24;
    *(_BYTE *)(a1 + 16) &= ~1u;
    *(_QWORD *)a1 = v16;
  }
  else
  {
    v17 = sub_2241E40();
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = v17;
  }
  return a1;
}
