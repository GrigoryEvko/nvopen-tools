// Function: sub_1AD4770
// Address: 0x1ad4770
//
_QWORD *__fastcall sub_1AD4770(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r15
  unsigned __int8 *v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rbx
  __int64 v11; // rdi
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 *v14; // rax
  _QWORD *result; // rax
  __int64 v16; // rax
  unsigned __int64 v17; // r15
  __int64 v18; // [rsp+8h] [rbp-A8h]
  __int64 v19; // [rsp+18h] [rbp-98h]
  __int64 v20; // [rsp+18h] [rbp-98h]
  unsigned __int8 *v21; // [rsp+28h] [rbp-88h] BYREF
  unsigned __int8 *v22; // [rsp+30h] [rbp-80h] BYREF
  __int64 v23; // [rsp+38h] [rbp-78h]
  __int64 v24; // [rsp+40h] [rbp-70h]
  _QWORD *v25; // [rsp+48h] [rbp-68h]
  __int64 v26; // [rsp+50h] [rbp-60h]
  int v27; // [rsp+58h] [rbp-58h]
  __int64 v28; // [rsp+60h] [rbp-50h]
  __int64 v29; // [rsp+68h] [rbp-48h]

  v19 = *(_QWORD *)(a4 + 48);
  v7 = *(_QWORD *)(*a2 + 24LL);
  v23 = a4;
  v22 = 0;
  v25 = (_QWORD *)sub_157E9C0(a4);
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v24 = v19;
  if ( v19 != v23 + 40 )
  {
    if ( !v19 )
      BUG();
    v8 = *(unsigned __int8 **)(v19 + 24);
    v21 = v8;
    if ( v8 )
    {
      sub_1623A60((__int64)&v21, (__int64)v8, 2);
      if ( v22 )
        sub_161E7C0((__int64)&v22, (__int64)v22);
      v22 = v21;
      if ( v21 )
        sub_1623210((__int64)&v21, v21, (__int64)&v22);
    }
  }
  v9 = a3;
  v10 = 1;
  v11 = sub_1632FA0(v9);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v7 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v16 = *(_QWORD *)(v7 + 32);
        v7 = *(_QWORD *)(v7 + 24);
        v10 *= v16;
        continue;
      case 1:
        v12 = 16;
        break;
      case 2:
        v12 = 32;
        break;
      case 3:
      case 9:
        v12 = 64;
        break;
      case 4:
        v12 = 80;
        break;
      case 5:
      case 6:
        v12 = 128;
        break;
      case 7:
        v12 = 8 * (unsigned int)sub_15A9520(v11, 0);
        break;
      case 0xB:
        v12 = *(_DWORD *)(v7 + 8) >> 8;
        break;
      case 0xD:
        v12 = 8LL * *(_QWORD *)sub_15A9930(v11, v7);
        break;
      case 0xE:
        v18 = *(_QWORD *)(v7 + 24);
        v20 = *(_QWORD *)(v7 + 32);
        v17 = (unsigned int)sub_15A9FE0(v11, v18);
        v12 = 8 * v20 * v17 * ((v17 + ((unsigned __int64)(sub_127FA20(v11, v18) + 7) >> 3) - 1) / v17);
        break;
      case 0xF:
        v12 = 8 * (unsigned int)sub_15A9520(v11, *(_DWORD *)(v7 + 8) >> 8);
        break;
    }
    break;
  }
  v13 = sub_1643360(v25);
  v14 = (__int64 *)sub_159C470(v13, (unsigned __int64)(v12 * v10 + 7) >> 3, 0);
  result = sub_15E7430((__int64 *)&v22, a1, 1u, a2, 1u, v14, 0, 0, 0, 0, 0);
  if ( v22 )
    return (_QWORD *)sub_161E7C0((__int64)&v22, (__int64)v22);
  return result;
}
