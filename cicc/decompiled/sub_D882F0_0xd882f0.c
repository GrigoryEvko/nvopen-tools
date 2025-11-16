// Function: sub_D882F0
// Address: 0xd882f0
//
__int64 *__fastcall sub_D882F0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r15
  __int64 v5; // r12
  char v6; // r14
  char *v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rdx
  unsigned int v10; // r12d
  __int64 v12; // rsi
  unsigned __int64 v13; // rsi
  __int64 v14; // r15
  bool v15; // al
  __int64 v16; // rsi
  unsigned int v17; // ecx
  __int64 v18; // rax
  unsigned int v19; // ebx
  bool v20; // al
  unsigned int v21; // eax
  unsigned int v22; // eax
  char *v23; // [rsp+8h] [rbp-88h]
  unsigned int v24; // [rsp+8h] [rbp-88h]
  unsigned __int64 v25; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v26; // [rsp+18h] [rbp-78h]
  __int64 v27; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v28; // [rsp+28h] [rbp-68h]
  unsigned __int64 v29; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v30; // [rsp+38h] [rbp-58h]
  char *v31; // [rsp+40h] [rbp-50h] BYREF
  __int64 v32; // [rsp+48h] [rbp-48h]
  __int64 v33[8]; // [rsp+50h] [rbp-40h] BYREF

  v3 = sub_B43CC0(a2);
  v4 = *(_QWORD *)(a2 + 72);
  v5 = v3;
  v6 = sub_AE5020(v3, v4);
  v7 = (char *)sub_9208B0(v5, v4);
  v8 = *(_QWORD *)(a2 + 8);
  v32 = v9;
  LOBYTE(v4) = v9;
  v23 = v7;
  v31 = v7;
  v10 = sub_AE43A0(v5, v8);
  sub_AADB10((__int64)a1, v10, 0);
  if ( !(_BYTE)v4 )
  {
    v26 = v10;
    v12 = (((unsigned __int64)(v23 + 7) >> 3) + (1LL << v6) - 1) >> v6 << v6;
    if ( v10 > 0x40 )
    {
      sub_C43690((__int64)&v25, v12, 1);
      v14 = 1LL << ((unsigned __int8)v26 - 1);
      if ( v26 > 0x40 )
      {
        if ( (*(_QWORD *)(v25 + 8LL * ((v26 - 1) >> 6)) & v14) != 0 )
          goto LABEL_8;
        v24 = v26;
        v15 = v24 == (unsigned int)sub_C444A0((__int64)&v25);
        goto LABEL_12;
      }
    }
    else
    {
      v13 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v10) & v12;
      if ( !v10 )
        v13 = 0;
      v14 = 1LL << ((unsigned __int8)v10 - 1);
      v25 = v13;
    }
    if ( (v14 & v25) != 0 )
    {
LABEL_8:
      sub_969240((__int64 *)&v25);
      return a1;
    }
    v15 = v25 == 0;
LABEL_12:
    if ( v15 )
      goto LABEL_8;
    if ( !(unsigned __int8)sub_B4CE70(a2) )
    {
LABEL_20:
      v30 = v26;
      if ( v26 > 0x40 )
        sub_C43780((__int64)&v29, (const void **)&v25);
      else
        v29 = v25;
      v28 = v10;
      if ( v10 > 0x40 )
        sub_C43690((__int64)&v27, 0, 0);
      else
        v27 = 0;
      sub_AADC30((__int64)&v31, (__int64)&v27, (__int64 *)&v29);
      sub_D859E0(a1, (__int64 *)&v31);
      sub_969240(v33);
      sub_969240((__int64 *)&v31);
      sub_969240(&v27);
      goto LABEL_18;
    }
    v16 = *(_QWORD *)(a2 - 32);
    if ( *(_BYTE *)v16 != 17 )
      goto LABEL_8;
    LOBYTE(v27) = 0;
    v17 = *(_DWORD *)(v16 + 32);
    v30 = v17;
    if ( v17 > 0x40 )
    {
      sub_C43780((__int64)&v29, (const void **)(v16 + 24));
      v19 = v30;
      v18 = 1LL << ((unsigned __int8)v30 - 1);
      if ( v30 > 0x40 )
      {
        if ( (*(_QWORD *)(v29 + 8LL * ((v30 - 1) >> 6)) & v18) != 0 )
          goto LABEL_18;
        v20 = v19 == (unsigned int)sub_C444A0((__int64)&v29);
LABEL_28:
        if ( v20 )
          goto LABEL_18;
        sub_C44B10((__int64)&v31, (char **)&v29, v10);
        if ( v30 > 0x40 && v29 )
          j_j___libc_free_0_0(v29);
        v29 = (unsigned __int64)v31;
        v21 = v32;
        LODWORD(v32) = 0;
        v30 = v21;
        sub_969240((__int64 *)&v31);
        sub_C4A7C0((__int64)&v31, (__int64)&v25, (__int64)&v29, (bool *)&v27);
        if ( v26 > 0x40 && v25 )
          j_j___libc_free_0_0(v25);
        v25 = (unsigned __int64)v31;
        v22 = v32;
        LODWORD(v32) = 0;
        v26 = v22;
        sub_969240((__int64 *)&v31);
        if ( (_BYTE)v27 )
          goto LABEL_18;
        sub_969240((__int64 *)&v29);
        goto LABEL_20;
      }
    }
    else
    {
      v29 = *(_QWORD *)(v16 + 24);
      v18 = 1LL << ((unsigned __int8)v17 - 1);
    }
    if ( (v29 & v18) != 0 )
    {
LABEL_18:
      sub_969240((__int64 *)&v29);
      sub_969240((__int64 *)&v25);
      return a1;
    }
    v20 = v29 == 0;
    goto LABEL_28;
  }
  return a1;
}
