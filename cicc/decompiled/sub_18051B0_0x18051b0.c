// Function: sub_18051B0
// Address: 0x18051b0
//
__int64 __fastcall sub_18051B0(__int64 a1, __int64 *a2, __int64 a3, char a4)
{
  __int64 v7; // r12
  _QWORD *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r12
  _QWORD *v11; // rax
  __int64 ***v12; // r12
  __int64 v13; // rsi
  __int64 v14; // rsi
  __int64 *v16; // rax
  __int64 *v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdi
  unsigned __int64 *v20; // r13
  __int64 **v21; // rax
  unsigned __int64 v22; // rcx
  __int64 v23; // rsi
  __int64 v24; // rsi
  unsigned __int8 *v25; // rsi
  __int64 v26; // [rsp+8h] [rbp-78h] BYREF
  __int64 v27; // [rsp+10h] [rbp-70h] BYREF
  __int16 v28; // [rsp+20h] [rbp-60h]
  __int64 v29[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v30; // [rsp+40h] [rbp-40h]

  v7 = *(_QWORD *)(a3 + 16);
  v8 = (_QWORD *)a2[3];
  v29[0] = (__int64)"MyAlloca";
  v30 = 259;
  if ( a4 )
  {
    v9 = sub_1643360(v8);
    v10 = sub_159C470(v9, v7, 0);
    v11 = (_QWORD *)sub_1643330((_QWORD *)a2[3]);
    v12 = (__int64 ***)sub_17CEAE0(a2, v11, v10, v29);
  }
  else
  {
    v16 = (__int64 *)sub_1643330(v8);
    v17 = sub_1645D80(v16, v7);
    v12 = (__int64 ***)sub_17CEAE0(a2, v17, 0, v29);
  }
  LODWORD(v13) = dword_4FA7E80;
  if ( (unsigned __int64)(unsigned int)dword_4FA7E80 < *(_QWORD *)(a3 + 8) )
    v13 = *(_QWORD *)(a3 + 8);
  sub_15F8A20((__int64)v12, v13);
  v14 = *(_QWORD *)(a1 + 488);
  v28 = 257;
  if ( (__int64 **)v14 != *v12 )
  {
    if ( *((_BYTE *)v12 + 16) > 0x10u )
    {
      v30 = 257;
      v18 = sub_15FDFF0((__int64)v12, v14, (__int64)v29, 0);
      v19 = a2[1];
      v12 = (__int64 ***)v18;
      if ( v19 )
      {
        v20 = (unsigned __int64 *)a2[2];
        sub_157E9D0(v19 + 40, v18);
        v21 = v12[3];
        v22 = *v20;
        v12[4] = (__int64 **)v20;
        v22 &= 0xFFFFFFFFFFFFFFF8LL;
        v12[3] = (__int64 **)(v22 | (unsigned __int8)v21 & 7);
        *(_QWORD *)(v22 + 8) = v12 + 3;
        *v20 = *v20 & 7 | (unsigned __int64)(v12 + 3);
      }
      sub_164B780((__int64)v12, &v27);
      v23 = *a2;
      if ( *a2 )
      {
        v26 = *a2;
        sub_1623A60((__int64)&v26, v23, 2);
        v24 = (__int64)v12[6];
        if ( v24 )
          sub_161E7C0((__int64)(v12 + 6), v24);
        v25 = (unsigned __int8 *)v26;
        v12[6] = (__int64 **)v26;
        if ( v25 )
          sub_1623210((__int64)&v26, v25, (__int64)(v12 + 6));
      }
    }
    else
    {
      return sub_15A4A70(v12, v14);
    }
  }
  return (__int64)v12;
}
