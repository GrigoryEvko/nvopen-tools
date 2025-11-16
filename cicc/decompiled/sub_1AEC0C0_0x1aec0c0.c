// Function: sub_1AEC0C0
// Address: 0x1aec0c0
//
void __fastcall sub_1AEC0C0(__int64 a1, __int64 a2, unsigned int *a3, __int64 a4)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r13
  unsigned __int64 v9; // r15
  __int64 v10; // rdi
  int v11; // r13d
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  _BYTE *v21; // [rsp+8h] [rbp-88h]
  _BYTE *v22; // [rsp+10h] [rbp-80h] BYREF
  __int64 v23; // [rsp+18h] [rbp-78h]
  _BYTE v24[112]; // [rsp+20h] [rbp-70h] BYREF

  v22 = v24;
  v23 = 0x400000000LL;
  sub_1624960(a1, a3, a4);
  if ( *(__int16 *)(a1 + 18) < 0 )
    sub_161F980(a1, (__int64)&v22);
  v7 = (__int64)v22;
  v8 = 16LL * (unsigned int)v23;
  v21 = &v22[v8];
  if ( &v22[v8] != v22 )
  {
    v9 = (unsigned __int64)v22;
    do
    {
      v10 = *(_QWORD *)(a2 + 48);
      v11 = *(_DWORD *)v9;
      if ( v10 || *(__int16 *)(a2 + 18) < 0 )
        v10 = sub_1625790(a2, v11);
      v12 = *(_QWORD *)(v9 + 8);
      switch ( v11 )
      {
        case 1:
          v19 = sub_14A8140(v10, v12);
          sub_1625C10(a1, 1, v19);
          break;
        case 3:
          v18 = sub_161F2A0(v10, v12, v6, v7);
          sub_1625C10(a1, 3, v18);
          break;
        case 4:
          v17 = sub_1628300(v10, v12);
          sub_1625C10(a1, 4, v17);
          break;
        case 6:
          sub_1625C10(a1, 6, v10);
          break;
        case 7:
          v16 = sub_1631A90(v10, v12);
          sub_1625C10(a1, 7, v16);
          break;
        case 8:
        case 10:
          v15 = sub_1630FC0(v10, v12);
          sub_1625C10(a1, v11, v15);
          break;
        case 11:
          sub_1625C10(a1, 11, v10);
          break;
        case 12:
        case 13:
          v14 = sub_161F460(v10, v12);
          sub_1625C10(a1, v11, v14);
          break;
        case 16:
          break;
        case 17:
          v20 = sub_161F460(v10, v12);
          sub_1625C10(a1, 17, v20);
          break;
        default:
          sub_1625C10(a1, v11, 0);
          break;
      }
      v9 += 16LL;
    }
    while ( v21 != (_BYTE *)v9 );
  }
  if ( *(_QWORD *)(a2 + 48) || *(__int16 *)(a2 + 18) < 0 )
  {
    v13 = sub_1625790(a2, 16);
    if ( v13 )
    {
      if ( (unsigned __int8)(*(_BYTE *)(a1 + 16) - 54) <= 1u )
        sub_1625C10(a1, 16, v13);
    }
  }
  if ( v22 != v24 )
    _libc_free((unsigned __int64)v22);
}
