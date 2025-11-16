// Function: sub_1917860
// Address: 0x1917860
//
__int64 __fastcall sub_1917860(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r12
  unsigned int v7; // eax
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned int v11; // eax
  int v12; // eax
  _DWORD *v13; // rdx
  __int64 v14; // rax
  _QWORD *v15; // rax
  __int64 v16; // rsi
  unsigned __int64 v17; // rcx
  __int64 v18; // rsi
  unsigned __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // rsi
  __int64 v23; // r8
  unsigned __int8 *v24; // rsi
  int v25; // ebx
  __int64 v28; // [rsp+10h] [rbp-70h]
  __int64 v29; // [rsp+18h] [rbp-68h]
  _QWORD v30[2]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v31[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v32; // [rsp+40h] [rbp-40h]

  v6 = a1 + 152;
  v7 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( v7 )
  {
    v8 = 0;
    v29 = 24LL * v7;
    do
    {
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v9 = *(_QWORD *)(a2 - 8);
      else
        v9 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v10 = *(_QWORD *)(v9 + v8);
      if ( *(_BYTE *)(v10 + 16) > 0x11u )
      {
        v28 = *(_QWORD *)(v9 + v8);
        if ( !(unsigned __int8)sub_190ABB0(v6, v10) )
          return 0;
        v11 = sub_190AC30(v6, v28, 1);
        v12 = sub_19170B0(v6, a3, a4, v11);
        v13 = sub_1910330(a1, a3, v12);
        if ( !v13 )
          return 0;
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v14 = *(_QWORD *)(a2 - 8);
        else
          v14 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        v15 = (_QWORD *)(v8 + v14);
        if ( *v15 )
        {
          v16 = v15[1];
          v17 = v15[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v17 = v16;
          if ( v16 )
            *(_QWORD *)(v16 + 16) = *(_QWORD *)(v16 + 16) & 3LL | v17;
        }
        *v15 = v13;
        v18 = *((_QWORD *)v13 + 1);
        v15[1] = v18;
        if ( v18 )
          *(_QWORD *)(v18 + 16) = (unsigned __int64)(v15 + 1) | *(_QWORD *)(v18 + 16) & 3LL;
        v15[2] = v15[2] & 3LL | (unsigned __int64)(v13 + 2);
        *((_QWORD *)v13 + 1) = v15;
      }
      v8 += 24;
    }
    while ( v29 != v8 );
  }
  v19 = sub_157EBA0(a3);
  sub_15F2120(a2, v19);
  v30[0] = sub_1649960(a2);
  v32 = 773;
  v31[0] = (__int64)v30;
  v30[1] = v20;
  v31[1] = (__int64)".pre";
  sub_164B780(a2, v31);
  v21 = *(_QWORD *)(a2 + 48);
  v31[0] = v21;
  if ( v21 )
  {
    sub_1623A60((__int64)v31, v21, 2);
    v22 = *(_QWORD *)(a2 + 48);
    v23 = a2 + 48;
    if ( v22 )
    {
      sub_161E7C0(a2 + 48, v22);
      v24 = (unsigned __int8 *)v31[0];
      v23 = a2 + 48;
    }
    else
    {
      v24 = (unsigned __int8 *)v31[0];
    }
    *(_QWORD *)(a2 + 48) = v24;
    if ( v24 )
      sub_1623210((__int64)v31, v24, v23);
  }
  v25 = sub_1911FD0(v6, a2);
  sub_19110A0(v6, a2, v25);
  sub_1910810(a1, v25, a2, a3);
  return 1;
}
