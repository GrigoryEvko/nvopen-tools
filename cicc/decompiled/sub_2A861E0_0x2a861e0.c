// Function: sub_2A861E0
// Address: 0x2a861e0
//
__int64 __fastcall sub_2A861E0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rbx
  unsigned __int64 v4; // rax
  __int64 *v5; // rsi
  unsigned __int64 v6; // rdi
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // r14
  _QWORD *v11; // rdi
  __int64 *v12; // rbx
  __int64 v13; // r14
  _QWORD *v14; // rdi
  const char *v15; // r14
  unsigned __int16 v16; // r15
  _QWORD *v17; // rdi
  unsigned int v18; // r12d
  __int64 *v20; // [rsp+8h] [rbp-88h]
  __int64 *v21; // [rsp+10h] [rbp-80h] BYREF
  __int64 *v22; // [rsp+18h] [rbp-78h]
  __int64 *v23; // [rsp+20h] [rbp-70h]
  const char *v24; // [rsp+30h] [rbp-60h] BYREF
  __int64 v25; // [rsp+38h] [rbp-58h]
  char v26; // [rsp+50h] [rbp-40h]
  char v27; // [rsp+51h] [rbp-3Fh]

  v1 = a1 + 72;
  v2 = *(_QWORD *)(a1 + 80);
  v21 = 0;
  v22 = 0;
  v23 = 0;
  if ( v2 == a1 + 72 )
    return 0;
  do
  {
    while ( 1 )
    {
      if ( !v2 )
        BUG();
      v4 = *(_QWORD *)(v2 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v4 == v2 + 24 )
        goto LABEL_32;
      if ( !v4 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v4 - 24) - 30 > 0xA )
LABEL_32:
        BUG();
      if ( *(_BYTE *)(v4 - 24) != 36 )
        goto LABEL_3;
      v5 = v22;
      v24 = (const char *)(v2 - 24);
      if ( v22 != v23 )
        break;
      sub_F38A10((__int64)&v21, v22, &v24);
LABEL_3:
      v2 = *(_QWORD *)(v2 + 8);
      if ( v1 == v2 )
        goto LABEL_13;
    }
    if ( v22 )
    {
      *v22 = v2 - 24;
      v5 = v22;
    }
    v22 = v5 + 1;
    v2 = *(_QWORD *)(v2 + 8);
  }
  while ( v1 != v2 );
LABEL_13:
  v6 = (unsigned __int64)v21;
  if ( (unsigned __int64)((char *)v22 - (char *)v21) <= 8 )
  {
    v18 = 0;
  }
  else
  {
    v27 = 1;
    v24 = "UnifiedUnreachableBlock";
    v26 = 3;
    v7 = sub_B2BE50(a1);
    v8 = sub_22077B0(0x50u);
    v9 = v8;
    if ( v8 )
      sub_AA4D50(v8, v7, (__int64)&v24, a1, 0);
    v10 = sub_B2BE50(a1);
    sub_B43C20((__int64)&v24, v9);
    v11 = sub_BD2C40(72, unk_3F148B8);
    if ( v11 )
      sub_B4C8A0((__int64)v11, v10, (__int64)v24, v25);
    v6 = (unsigned __int64)v21;
    v12 = v21;
    v20 = v22;
    if ( v22 != v21 )
    {
      do
      {
        v13 = *v12;
        v14 = (_QWORD *)((*(_QWORD *)(*v12 + 48) & 0xFFFFFFFFFFFFFFF8LL) - 24);
        if ( (*(_QWORD *)(*v12 + 48) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          v14 = 0;
        sub_B43D60(v14);
        sub_B43C20((__int64)&v24, v13);
        v15 = v24;
        v16 = v25;
        v17 = sub_BD2C40(72, 1u);
        if ( v17 )
          sub_B4C8F0((__int64)v17, v9, 1u, (__int64)v15, v16);
        ++v12;
      }
      while ( v20 != v12 );
      v6 = (unsigned __int64)v21;
    }
    v18 = 1;
  }
  if ( v6 )
    j_j___libc_free_0(v6);
  return v18;
}
