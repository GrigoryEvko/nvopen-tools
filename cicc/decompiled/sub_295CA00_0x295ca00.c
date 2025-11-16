// Function: sub_295CA00
// Address: 0x295ca00
//
void __fastcall sub_295CA00(__int64 *a1, __int64 a2, _BYTE *a3)
{
  _BYTE *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdi
  bool v13; // zf
  unsigned __int8 v14; // al
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // r8
  __int64 v18; // rcx
  _BYTE *v19; // rdi
  __int64 v20; // r15
  _QWORD *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rdi
  __int64 v28; // [rsp+8h] [rbp-88h] BYREF
  __int64 v29; // [rsp+10h] [rbp-80h] BYREF
  __int64 v30[3]; // [rsp+18h] [rbp-78h] BYREF
  char v31; // [rsp+30h] [rbp-60h]
  char v32; // [rsp+58h] [rbp-38h]

  v4 = sub_2958930(a3);
  if ( *v4 <= 0x15u )
    return;
  v7 = (__int64)v4;
  if ( (unsigned __int8)sub_D48480(*a1, (__int64)v4, v5, v6) )
  {
    v27 = a1[1];
    v29 = a2;
    v30[0] = v7 & 0xFFFFFFFFFFFFFFFBLL;
    v31 = 0;
    v32 = 0;
    sub_2958F30(v27, (unsigned __int64)&v29, v8, v9, v10, v11);
    sub_295C970(v30);
    return;
  }
  if ( *(_BYTE *)v7 <= 0x1Cu )
    return;
  v12 = *(_QWORD *)(v7 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17 <= 1 )
    v12 = **(_QWORD **)(v12 + 16);
  v13 = !sub_BCAC40(v12, 1);
  v14 = *(_BYTE *)v7;
  if ( v13 )
    goto LABEL_14;
  if ( v14 == 57 )
    goto LABEL_24;
  if ( v14 == 86 )
  {
    v15 = *(_QWORD *)(v7 + 8);
    if ( *(_QWORD *)(*(_QWORD *)(v7 - 96) + 8LL) != v15 || **(_BYTE **)(v7 - 32) > 0x15u )
      goto LABEL_16;
    if ( !sub_AC30F0(*(_QWORD *)(v7 - 32)) )
    {
      v14 = *(_BYTE *)v7;
      goto LABEL_14;
    }
LABEL_24:
    sub_295B990(&v28, *a1, v7);
    if ( (v28 & 0xFFFFFFFFFFFFFFF8LL) != 0 && ((v28 & 4) == 0 || *(_DWORD *)((v28 & 0xFFFFFFFFFFFFFFF8LL) + 8)) )
    {
      v20 = a1[1];
      v21 = sub_295C9D0(&v28);
      v29 = a2;
      sub_295C880((unsigned __int64 *)v30, v21, v22);
      v31 = 0;
      v32 = 0;
      sub_2958F30(v20, (unsigned __int64)&v29, v23, v24, v25, v26);
      sub_295C970(v30);
    }
    sub_295C970(&v28);
    return;
  }
LABEL_14:
  if ( v14 <= 0x1Cu )
    return;
  v15 = *(_QWORD *)(v7 + 8);
LABEL_16:
  if ( (unsigned int)*(unsigned __int8 *)(v15 + 8) - 17 <= 1 )
    v15 = **(_QWORD **)(v15 + 16);
  if ( sub_BCAC40(v15, 1) )
  {
    if ( *(_BYTE *)v7 == 58 )
      goto LABEL_24;
    if ( *(_BYTE *)v7 == 86 )
    {
      v18 = *(_QWORD *)(v7 + 8);
      if ( *(_QWORD *)(*(_QWORD *)(v7 - 96) + 8LL) == v18 )
      {
        v19 = *(_BYTE **)(v7 - 64);
        if ( *v19 <= 0x15u && sub_AD7A80(v19, 1, v16, v18, v17) )
          goto LABEL_24;
      }
    }
  }
}
