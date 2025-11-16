// Function: sub_1216280
// Address: 0x1216280
//
__int64 __fastcall sub_1216280(
        __int64 a1,
        void (__fastcall *a2)(_QWORD *, __int64, _QWORD, _QWORD, _BYTE *, __int64),
        __int64 a3)
{
  __int64 v4; // rax
  int v5; // eax
  unsigned int v6; // r13d
  char v7; // al
  _QWORD *v8; // rsi
  __int64 v10; // rax
  _QWORD *v11; // [rsp+10h] [rbp-2B0h] BYREF
  __int64 v12; // [rsp+18h] [rbp-2A8h] BYREF
  _BYTE *v13; // [rsp+20h] [rbp-2A0h] BYREF
  __int64 v14; // [rsp+28h] [rbp-298h]
  _QWORD v15[2]; // [rsp+30h] [rbp-290h] BYREF
  _QWORD v16[2]; // [rsp+40h] [rbp-280h] BYREF
  __int64 v17; // [rsp+50h] [rbp-270h] BYREF
  _QWORD *v18; // [rsp+60h] [rbp-260h] BYREF
  __int16 v19; // [rsp+80h] [rbp-240h]
  _QWORD v20[2]; // [rsp+90h] [rbp-230h] BYREF
  __int64 v21; // [rsp+A0h] [rbp-220h] BYREF
  char v22; // [rsp+B0h] [rbp-210h]
  char v23; // [rsp+280h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 344);
  v13 = v15;
  sub_12060D0((__int64 *)&v13, *(_BYTE **)(v4 + 760), *(_QWORD *)(v4 + 760) + *(_QWORD *)(v4 + 768));
  v11 = 0;
  while ( 1 )
  {
    while ( 1 )
    {
      v5 = *(_DWORD *)(a1 + 240);
      if ( v5 != 63 )
        break;
      if ( (unsigned __int8)sub_120B4E0(a1, (__int64)&v13, &v11) )
        goto LABEL_14;
    }
    if ( v5 != 65 )
      break;
    if ( (unsigned __int8)sub_120B790(a1) )
    {
LABEL_14:
      v6 = 1;
      goto LABEL_10;
    }
  }
  a2(v20, a3, *(_QWORD *)(*(_QWORD *)(a1 + 344) + 232LL), *(_QWORD *)(*(_QWORD *)(a1 + 344) + 240LL), v13, v14);
  if ( v22 )
  {
    sub_2240AE0(&v13, v20);
    v11 = 0;
    if ( v22 )
    {
      v22 = 0;
      if ( (__int64 *)v20[0] != &v21 )
        j_j___libc_free_0(v20[0], v21 + 1);
    }
  }
  sub_AE41B0((__int64)v20, v13, v14);
  v6 = v23 & 1;
  v7 = (2 * v6) | v23 & 0xFD;
  v23 = v7;
  if ( (_BYTE)v6 )
  {
    v23 = v7 & 0xFD;
    v10 = v20[0];
    v20[0] = 0;
    v12 = v10 | 1;
    sub_C64870((__int64)v16, &v12);
    v8 = v11;
    v19 = 260;
    v18 = v16;
    sub_11FD800(a1 + 176, (unsigned __int64)v11, (__int64)&v18, 1);
    if ( (__int64 *)v16[0] != &v17 )
    {
      v8 = (_QWORD *)(v17 + 1);
      j_j___libc_free_0(v16[0], v17 + 1);
    }
    v6 = (v12 & 1) == 0;
    if ( (v12 & 1) != 0 || (v12 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v12, (__int64)v8);
  }
  else
  {
    v8 = v20;
    sub_BA9570(*(_QWORD *)(a1 + 344), (__int64)v20);
  }
  if ( (v23 & 2) != 0 )
    sub_9D2AF0(v20);
  if ( (v23 & 1) != 0 )
  {
    if ( v20[0] )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v20[0] + 8LL))(v20[0]);
  }
  else
  {
    sub_AE4030(v20, (__int64)v8);
  }
LABEL_10:
  if ( v13 != (_BYTE *)v15 )
    j_j___libc_free_0(v13, v15[0] + 1LL);
  return v6;
}
