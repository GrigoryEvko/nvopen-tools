// Function: sub_1CA18D0
// Address: 0x1ca18d0
//
_BYTE *__fastcall sub_1CA18D0(__int64 a1, _BYTE *a2, int a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 **v8; // r14
  unsigned __int8 v9; // al
  __int64 v10; // r12
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned __int8 *v13; // rsi
  __int64 v14; // rax
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 *v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rsi
  unsigned __int8 *v22; // rsi
  unsigned __int8 *v23; // [rsp+8h] [rbp-C8h] BYREF
  __int64 v24; // [rsp+10h] [rbp-C0h] BYREF
  __int16 v25; // [rsp+20h] [rbp-B0h]
  unsigned __int8 *v26[2]; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v27; // [rsp+40h] [rbp-90h]
  unsigned __int8 *v28; // [rsp+50h] [rbp-80h] BYREF
  __int64 v29; // [rsp+58h] [rbp-78h]
  __int64 *v30; // [rsp+60h] [rbp-70h]
  __int64 v31; // [rsp+68h] [rbp-68h]
  __int64 v32; // [rsp+70h] [rbp-60h]
  int v33; // [rsp+78h] [rbp-58h]
  __int64 v34; // [rsp+80h] [rbp-50h]
  __int64 v35; // [rsp+88h] [rbp-48h]

  v7 = sub_1646BA0(*(__int64 **)(*(_QWORD *)a2 + 24LL), a3);
  if ( v7 == *(_QWORD *)a2 )
    return a2;
  v8 = (__int64 **)v7;
  v9 = a2[16];
  v10 = (__int64)a2;
  if ( v9 == 72 )
  {
    v10 = *((_QWORD *)a2 - 3);
    if ( v8 == *(__int64 ***)v10 )
      return (_BYTE *)v10;
    v9 = *(_BYTE *)(v10 + 16);
  }
  if ( v9 <= 0x17u )
  {
    v16 = *(_QWORD *)(a4 + 80);
    if ( v16 )
      v16 -= 24;
    v11 = sub_157ED20(v16);
  }
  else
  {
    v11 = sub_15F3430(v10);
  }
  v12 = sub_16498A0(v11);
  v34 = 0;
  v35 = 0;
  v13 = *(unsigned __int8 **)(v11 + 48);
  v31 = v12;
  v33 = 0;
  v14 = *(_QWORD *)(v11 + 40);
  v28 = 0;
  v29 = v14;
  v32 = 0;
  v30 = (__int64 *)(v11 + 24);
  v26[0] = v13;
  if ( v13 )
  {
    sub_1623A60((__int64)v26, (__int64)v13, 2);
    if ( v28 )
      sub_161E7C0((__int64)&v28, (__int64)v28);
    v28 = v26[0];
    if ( v26[0] )
      sub_1623210((__int64)v26, v26[0], (__int64)&v28);
  }
  v25 = 257;
  if ( v8 != *(__int64 ***)v10 )
  {
    if ( *(_BYTE *)(v10 + 16) > 0x10u )
    {
      v27 = 257;
      v17 = sub_15FDBD0(48, v10, (__int64)v8, (__int64)v26, 0);
      v10 = v17;
      if ( v29 )
      {
        v18 = v30;
        sub_157E9D0(v29 + 40, v17);
        v19 = *(_QWORD *)(v10 + 24);
        v20 = *v18;
        *(_QWORD *)(v10 + 32) = v18;
        v20 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v10 + 24) = v20 | v19 & 7;
        *(_QWORD *)(v20 + 8) = v10 + 24;
        *v18 = *v18 & 7 | (v10 + 24);
      }
      sub_164B780(v10, &v24);
      if ( v28 )
      {
        v23 = v28;
        sub_1623A60((__int64)&v23, (__int64)v28, 2);
        v21 = *(_QWORD *)(v10 + 48);
        if ( v21 )
          sub_161E7C0(v10 + 48, v21);
        v22 = v23;
        *(_QWORD *)(v10 + 48) = v23;
        if ( v22 )
          sub_1623210((__int64)&v23, v22, v10 + 48);
      }
    }
    else
    {
      v10 = sub_15A46C0(48, (__int64 ***)v10, v8, 0);
    }
  }
  if ( *(_BYTE *)(v10 + 16) > 0x17u )
    sub_1CA1000(a1, (__int64)a2, v10);
  if ( v28 )
    sub_161E7C0((__int64)&v28, (__int64)v28);
  return (_BYTE *)v10;
}
