// Function: sub_3255900
// Address: 0x3255900
//
void __fastcall sub_3255900(__int64 a1, __int64 a2, char *a3)
{
  signed __int64 v4; // r8
  _BYTE *v5; // rcx
  size_t v6; // r13
  _BYTE *v7; // rax
  size_t v8; // rdx
  _BYTE *v9; // r13
  __int64 v10; // rdi
  __int64 v11; // r13
  __int64 v12; // rdx
  _QWORD *v13; // [rsp+10h] [rbp-120h] BYREF
  size_t v14; // [rsp+18h] [rbp-118h]
  _BYTE v15[16]; // [rsp+20h] [rbp-110h] BYREF
  const char *v16[4]; // [rsp+30h] [rbp-100h] BYREF
  __int16 v17; // [rsp+50h] [rbp-E0h]
  const char *v18; // [rsp+60h] [rbp-D0h] BYREF
  const char *v19; // [rsp+68h] [rbp-C8h]
  __int64 v20; // [rsp+70h] [rbp-C0h]
  _BYTE v21[184]; // [rsp+78h] [rbp-B8h] BYREF

  v13 = v15;
  v14 = 0;
  v15[0] = 0;
  sub_2241490((unsigned __int64 *)&v13, "caml", 4u);
  v4 = *(_QWORD *)(a1 + 176);
  v5 = *(_BYTE **)(a1 + 168);
  v6 = v14;
  if ( v4 >> 2 > 0 )
  {
    v7 = *(_BYTE **)(a1 + 168);
    while ( *v7 != 46 )
    {
      if ( v7[1] == 46 )
      {
        v4 = v7 + 1 - v5;
        goto LABEL_9;
      }
      if ( v7[2] == 46 )
      {
        v4 = v7 + 2 - v5;
        goto LABEL_9;
      }
      if ( v7[3] == 46 )
      {
        v4 = v7 + 3 - v5;
        goto LABEL_9;
      }
      v7 += 4;
      if ( &v5[4 * (v4 >> 2)] == v7 )
      {
        v12 = &v5[v4] - v7;
        goto LABEL_20;
      }
    }
    goto LABEL_8;
  }
  v12 = *(_QWORD *)(a1 + 176);
  v7 = *(_BYTE **)(a1 + 168);
LABEL_20:
  if ( v12 != 2 )
  {
    if ( v12 != 3 )
    {
      if ( v12 != 1 )
        goto LABEL_9;
      goto LABEL_23;
    }
    if ( *v7 == 46 )
    {
LABEL_8:
      v4 = v7 - v5;
      goto LABEL_9;
    }
    ++v7;
  }
  if ( *v7 == 46 )
    goto LABEL_8;
  ++v7;
LABEL_23:
  if ( *v7 == 46 )
    v4 = v7 - v5;
LABEL_9:
  sub_2241130((unsigned __int64 *)&v13, v14, 0, v5, v4);
  if ( v14 == 0x3FFFFFFFFFFFFFFFLL
    || v14 == 4611686018427387902LL
    || (sub_2241490((unsigned __int64 *)&v13, "__", 2u), v8 = strlen(a3), v8 > 0x3FFFFFFFFFFFFFFFLL - v14) )
  {
    sub_4262D8((__int64)"basic_string::append");
  }
  sub_2241490((unsigned __int64 *)&v13, a3, v8);
  v9 = (char *)v13 + v6;
  *v9 = toupper((char)*v9);
  v17 = 260;
  v18 = v21;
  v19 = 0;
  v20 = 128;
  v16[0] = (const char *)&v13;
  sub_E405D0((__int64)&v18, (char *)v16, a1 + 312);
  v10 = *(_QWORD *)(a2 + 216);
  v17 = 261;
  v16[0] = v18;
  v16[1] = v19;
  v11 = sub_E6C460(v10, v16);
  (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a2 + 224) + 296LL))(*(_QWORD *)(a2 + 224), v11, 9);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a2 + 224) + 208LL))(*(_QWORD *)(a2 + 224), v11, 0);
  if ( v18 != v21 )
    _libc_free((unsigned __int64)v18);
  if ( v13 != (_QWORD *)v15 )
    j_j___libc_free_0((unsigned __int64)v13);
}
