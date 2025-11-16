// Function: sub_14A6690
// Address: 0x14a6690
//
__int64 __fastcall sub_14A6690(__int64 a1, __int64 a2)
{
  __int64 *v2; // r13
  __int64 v3; // rax
  __int64 *v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rax
  unsigned __int64 v7; // r8
  unsigned __int64 v8; // r12
  _BYTE *v9; // rdx
  int v10; // ecx
  _BYTE *v11; // r9
  __int64 *v12; // rax
  __int64 v13; // rcx
  unsigned __int8 v14; // al
  unsigned int v15; // ebx
  __int64 v16; // rbx
  __int64 **v17; // rax
  __int64 *v18; // r12
  __int64 v19; // rax
  int v20; // eax
  char v21; // al
  __int64 v22; // rbx
  _BYTE *v23; // rsi
  char v24; // al
  unsigned int v25; // r12d
  char v27; // al
  __int64 v28; // [rsp+0h] [rbp-70h]
  __int64 v29; // [rsp+8h] [rbp-68h]
  unsigned __int64 v30; // [rsp+8h] [rbp-68h]
  _BYTE *v31; // [rsp+8h] [rbp-68h]
  _BYTE *v32; // [rsp+10h] [rbp-60h] BYREF
  __int64 v33; // [rsp+18h] [rbp-58h]
  _BYTE v34[80]; // [rsp+20h] [rbp-50h] BYREF

  v2 = (__int64 *)(a1 + 8);
  v3 = 3LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v4 = *(__int64 **)(a2 - 8);
    v5 = (__int64)&v4[v3];
  }
  else
  {
    v4 = (__int64 *)(a2 - v3 * 8);
    v5 = a2;
  }
  v6 = v5 - (_QWORD)v4;
  v32 = v34;
  v33 = 0x400000000LL;
  v7 = 0xAAAAAAAAAAAAAAABLL * (v6 >> 3);
  v8 = v7;
  if ( (unsigned __int64)v6 > 0x60 )
  {
    v28 = v6;
    v30 = 0xAAAAAAAAAAAAAAABLL * (v6 >> 3);
    sub_16CD150(&v32, v34, v30, 8);
    v11 = v32;
    v10 = v33;
    LODWORD(v7) = v30;
    v6 = v28;
    v9 = &v32[8 * (unsigned int)v33];
  }
  else
  {
    v9 = v34;
    v10 = 0;
    v11 = v34;
  }
  if ( v6 > 0 )
  {
    v12 = v4;
    do
    {
      v13 = *v12;
      v9 += 8;
      v12 += 3;
      *((_QWORD *)v9 - 1) = v13;
      --v8;
    }
    while ( v8 );
    v11 = v32;
    v10 = v33;
  }
  v14 = *(_BYTE *)(a2 + 16);
  v15 = v7 + v10;
  LODWORD(v33) = v7 + v10;
  if ( v14 > 0x17u )
  {
    if ( v14 == 77 || v14 == 86 )
      goto LABEL_33;
    if ( v14 != 53 )
      goto LABEL_19;
    v31 = v11;
    v27 = sub_15F8F00(a2);
    v11 = v31;
    if ( v27 )
      goto LABEL_32;
    v14 = *(_BYTE *)(a2 + 16);
    if ( v14 > 0x17u )
    {
LABEL_19:
      if ( v14 == 56 )
        goto LABEL_12;
LABEL_20:
      v20 = sub_14A4C90((__int64)v2, a2);
      goto LABEL_21;
    }
  }
  if ( v14 != 5 || *(_WORD *)(a2 + 18) != 32 )
    goto LABEL_20;
LABEL_12:
  v16 = v15 - 1LL;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v17 = *(__int64 ***)(a2 - 8);
  else
    v17 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v18 = *v17;
  v29 = (__int64)(v11 + 8);
  v19 = sub_16348C0(a2);
  v20 = sub_14A1310(v2, v19, v18, v29, v16);
LABEL_21:
  if ( !v20 )
  {
LABEL_32:
    v11 = v32;
LABEL_33:
    v25 = 0;
    goto LABEL_34;
  }
  v21 = *(_BYTE *)(a2 + 16);
  if ( v21 == 54 )
  {
    v11 = v32;
    v25 = 4;
    goto LABEL_34;
  }
  v22 = *(_QWORD *)a2;
  if ( v21 == 78 )
  {
    v23 = *(_BYTE **)(a2 - 24);
    if ( v23[16] || sub_14A2090((__int64)v2, v23) )
    {
      v11 = v32;
      v25 = 40;
      goto LABEL_34;
    }
    v24 = *(_BYTE *)(v22 + 8);
    if ( v24 != 13 )
      goto LABEL_29;
    v22 = **(_QWORD **)(v22 + 16);
  }
  v24 = *(_BYTE *)(v22 + 8);
LABEL_29:
  if ( v24 == 16 )
    v24 = *(_BYTE *)(*(_QWORD *)(v22 + 24) + 8LL);
  v11 = v32;
  v25 = (unsigned __int8)(v24 - 1) < 6u ? 3 : 1;
LABEL_34:
  if ( v11 != v34 )
    _libc_free((unsigned __int64)v11);
  return v25;
}
