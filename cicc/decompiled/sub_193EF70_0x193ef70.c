// Function: sub_193EF70
// Address: 0x193ef70
//
void __fastcall sub_193EF70(__int64 ***a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v7; // rbx
  __int64 v8; // rax
  unsigned __int8 *v9; // rsi
  __int64 v10; // rax
  __int64 **v11; // rdx
  __int64 v12; // rax
  __int64 *v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rsi
  unsigned __int8 *v17; // rsi
  unsigned __int8 *v18; // [rsp+8h] [rbp-C8h] BYREF
  __int64 v19; // [rsp+10h] [rbp-C0h] BYREF
  __int16 v20; // [rsp+20h] [rbp-B0h]
  unsigned __int8 *v21[2]; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v22; // [rsp+40h] [rbp-90h]
  unsigned __int8 *v23; // [rsp+50h] [rbp-80h] BYREF
  __int64 v24; // [rsp+58h] [rbp-78h]
  __int64 *v25; // [rsp+60h] [rbp-70h]
  __int64 v26; // [rsp+68h] [rbp-68h]
  __int64 v27; // [rsp+70h] [rbp-60h]
  int v28; // [rsp+78h] [rbp-58h]
  __int64 v29; // [rsp+80h] [rbp-50h]
  __int64 v30; // [rsp+88h] [rbp-48h]

  v7 = a2;
  if ( *(_BYTE *)(a2 + 16) == 77 )
    v7 = sub_193EB70(a2, (__int64)a1, a4, a5);
  v8 = sub_16498A0(v7);
  v29 = 0;
  v30 = 0;
  v9 = *(unsigned __int8 **)(v7 + 48);
  v26 = v8;
  v28 = 0;
  v10 = *(_QWORD *)(v7 + 40);
  v23 = 0;
  v24 = v10;
  v27 = 0;
  v25 = (__int64 *)(v7 + 24);
  v21[0] = v9;
  if ( v9 )
  {
    sub_1623A60((__int64)v21, (__int64)v9, 2);
    if ( v23 )
      sub_161E7C0((__int64)&v23, (__int64)v23);
    v23 = v21[0];
    if ( v21[0] )
      sub_1623210((__int64)v21, v21[0], (__int64)&v23);
  }
  v11 = *a1;
  v20 = 257;
  if ( v11 != *(__int64 ***)a3 )
  {
    if ( *(_BYTE *)(a3 + 16) > 0x10u )
    {
      v22 = 257;
      v12 = sub_15FDBD0(36, a3, (__int64)v11, (__int64)v21, 0);
      a3 = v12;
      if ( v24 )
      {
        v13 = v25;
        sub_157E9D0(v24 + 40, v12);
        v14 = *(_QWORD *)(a3 + 24);
        v15 = *v13;
        *(_QWORD *)(a3 + 32) = v13;
        v15 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(a3 + 24) = v15 | v14 & 7;
        *(_QWORD *)(v15 + 8) = a3 + 24;
        *v13 = *v13 & 7 | (a3 + 24);
      }
      sub_164B780(a3, &v19);
      if ( v23 )
      {
        v18 = v23;
        sub_1623A60((__int64)&v18, (__int64)v23, 2);
        v16 = *(_QWORD *)(a3 + 48);
        if ( v16 )
          sub_161E7C0(a3 + 48, v16);
        v17 = v18;
        *(_QWORD *)(a3 + 48) = v18;
        if ( v17 )
          sub_1623210((__int64)&v18, v17, a3 + 48);
      }
    }
    else
    {
      a3 = sub_15A46C0(36, (__int64 ***)a3, v11, 0);
    }
  }
  sub_1648780(a2, (__int64)a1, a3);
  if ( v23 )
    sub_161E7C0((__int64)&v23, (__int64)v23);
}
