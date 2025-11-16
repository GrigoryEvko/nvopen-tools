// Function: sub_1AC4D50
// Address: 0x1ac4d50
//
__int64 __fastcall sub_1AC4D50(__int64 a1, char a2)
{
  char *v2; // r14
  _QWORD *v4; // r15
  size_t v5; // rbx
  const char *v6; // rdi
  __int64 v7; // rax
  _QWORD *v8; // rdi
  __int64 v9; // rdx
  unsigned int v10; // r15d
  __int64 v11; // rdx
  __int64 v12; // rbx
  __int64 v14; // rdi
  unsigned __int64 v15; // rax
  __int64 v16; // r14
  _QWORD *v17; // rax
  _QWORD *v18; // rax
  unsigned __int8 *v19; // rsi
  unsigned __int8 *v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int8 *v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // rcx
  __int64 v26; // [rsp+0h] [rbp-80h]
  __int64 v27; // [rsp+8h] [rbp-78h]
  size_t v28; // [rsp+10h] [rbp-70h]
  char *v29; // [rsp+18h] [rbp-68h]
  __int64 v30; // [rsp+20h] [rbp-60h]
  __int64 v31; // [rsp+28h] [rbp-58h]
  unsigned __int8 *v32; // [rsp+38h] [rbp-48h] BYREF
  unsigned __int8 *v33; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int8 *v34[7]; // [rsp+48h] [rbp-38h] BYREF

  v2 = "instrument-function-entry-inlined";
  v4 = (_QWORD *)(a1 + 112);
  v28 = (-(__int64)(a2 == 0) & 0xFFFFFFFFFFFFFFF8LL) + 32;
  v5 = (-(__int64)(a2 == 0) & 0xFFFFFFFFFFFFFFF8LL) + 33;
  if ( !a2 )
    v2 = "instrument-function-entry";
  v6 = "instrument-function-exit";
  if ( a2 )
    v6 = "instrument-function-exit-inlined";
  v29 = (char *)v6;
  v34[0] = (unsigned __int8 *)sub_1560340(v4, -1, v2, v5);
  v7 = sub_155D8B0((__int64 *)v34);
  v8 = v4;
  v31 = v9;
  v10 = 0;
  v26 = v7;
  v34[0] = (unsigned __int8 *)sub_1560340(v8, -1, v29, v28);
  v27 = sub_155D8B0((__int64 *)v34);
  v30 = v11;
  if ( v31 )
  {
    v33 = 0;
    v22 = sub_1626D20(a1);
    if ( v22 )
    {
      sub_15C7110(v34, *(_DWORD *)(v22 + 28), 0, v22, 0);
      v33 = v34[0];
      if ( !v34[0] )
      {
LABEL_48:
        v24 = *(_QWORD *)(a1 + 80);
        if ( v24 )
          v24 -= 24;
        v25 = sub_157EE30(v24);
        if ( v25 )
          v25 -= 24;
        sub_1AC4610((__int64 ***)a1, v26, v31, v25, (__int64 *)v34);
        if ( v34[0] )
          sub_161E7C0((__int64)v34, (__int64)v34[0]);
        sub_15E0EA0(a1, -1, v2, v5);
        if ( v33 )
          sub_161E7C0((__int64)&v33, (__int64)v33);
        v10 = 1;
        goto LABEL_6;
      }
      sub_1623210((__int64)v34, v34[0], (__int64)&v33);
      v23 = v33;
    }
    else
    {
      v23 = v33;
    }
    v34[0] = v23;
    if ( v23 )
      sub_1623A60((__int64)v34, (__int64)v23, 2);
    goto LABEL_48;
  }
LABEL_6:
  if ( v30 )
  {
    v12 = *(_QWORD *)(a1 + 80);
    if ( a1 + 72 == v12 )
    {
LABEL_8:
      sub_15E0EA0(a1, -1, v29, v28);
      return v10;
    }
    while ( 1 )
    {
      v14 = v12 - 24;
      if ( !v12 )
        v14 = 0;
      v15 = sub_157EBA0(v14);
      v16 = v15;
      if ( *(_BYTE *)(v15 + 16) != 25 )
        goto LABEL_21;
      if ( *(_QWORD *)(*(_QWORD *)(v15 + 40) + 48LL) != v15 + 24 )
      {
        v17 = (_QWORD *)(*(_QWORD *)(v15 + 24) & 0xFFFFFFFFFFFFFFF8LL);
        if ( (*(_QWORD *)(v16 + 24) & 0xFFFFFFFFFFFFFFF8LL) != 0
          && (*((_BYTE *)v17 - 8) != 71
           || *(_QWORD **)(v17[2] + 48LL) != v17 && (v17 = (_QWORD *)(*v17 & 0xFFFFFFFFFFFFFFF8LL)) != 0) )
        {
          v18 = v17 - 3;
          if ( *((_BYTE *)v18 + 16) == 78 && (*((_WORD *)v18 + 9) & 3) == 2 )
            v16 = (__int64)v18;
        }
      }
      v19 = *(unsigned __int8 **)(v16 + 48);
      v32 = 0;
      v33 = v19;
      if ( !v19 )
        break;
      sub_1623A60((__int64)&v33, (__int64)v19, 2);
      v20 = v33;
      if ( !v33 )
        break;
      if ( !v32 )
      {
        v32 = v33;
LABEL_11:
        sub_1623A60((__int64)&v32, (__int64)v20, 2);
LABEL_12:
        if ( v33 )
          sub_161E7C0((__int64)&v33, (__int64)v33);
        v34[0] = v32;
        if ( v32 )
          sub_1623A60((__int64)v34, (__int64)v32, 2);
        goto LABEL_16;
      }
      sub_161E7C0((__int64)&v32, (__int64)v32);
      v20 = v33;
      v32 = v33;
      if ( v33 )
        goto LABEL_11;
      v34[0] = 0;
LABEL_16:
      sub_1AC4610((__int64 ***)a1, v27, v30, v16, (__int64 *)v34);
      if ( v34[0] )
        sub_161E7C0((__int64)v34, (__int64)v34[0]);
      if ( v32 )
        sub_161E7C0((__int64)&v32, (__int64)v32);
      v10 = 1;
LABEL_21:
      v12 = *(_QWORD *)(v12 + 8);
      if ( a1 + 72 == v12 )
        goto LABEL_8;
    }
    v21 = sub_1626D20(a1);
    if ( v21 )
    {
      sub_15C7110(v34, 0, 0, v21, 0);
      if ( v32 )
        sub_161E7C0((__int64)&v32, (__int64)v32);
      v32 = v34[0];
      if ( v34[0] )
        sub_1623210((__int64)v34, v34[0], (__int64)&v32);
    }
    goto LABEL_12;
  }
  return v10;
}
