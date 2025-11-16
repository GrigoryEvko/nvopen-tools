// Function: sub_38F23C0
// Address: 0x38f23c0
//
__int64 __fastcall sub_38F23C0(__int64 a1, char a2)
{
  __int64 v3; // r12
  __int64 v4; // rdx
  unsigned int v5; // ecx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rdi
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned int v11; // r13d
  _DWORD *v13; // rax
  __int64 v14; // r8
  __int64 *v15; // r8
  __int64 v16; // rdx
  __int64 v17; // rdx
  int v18; // eax
  const char *v19; // rax
  char v20; // al
  __int64 *v21; // rdi
  __int64 v22; // rcx
  __int64 v23; // rax
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rax
  __int64 v26; // [rsp+8h] [rbp-88h]
  __int64 v27; // [rsp+10h] [rbp-80h]
  __int64 v28; // [rsp+18h] [rbp-78h]
  __int64 *v29; // [rsp+18h] [rbp-78h]
  __int64 v30; // [rsp+20h] [rbp-70h] BYREF
  signed __int64 v31; // [rsp+28h] [rbp-68h] BYREF
  __int64 v32[2]; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v33[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v34; // [rsp+50h] [rbp-40h]

  if ( !*(_BYTE *)(a1 + 845) && (unsigned __int8)sub_38E36C0(a1) )
    return 1;
  v32[0] = 0;
  v32[1] = 0;
  v3 = sub_3909290(a1 + 144);
  if ( (unsigned __int8)sub_38F0EE0(a1, v32, v4, v5) )
  {
    v33[0] = "expected identifier in directive";
    v34 = 259;
    return (unsigned int)sub_3909CF0(a1, v33, 0, 0, v6, v7);
  }
  v33[0] = v32;
  v8 = *(_QWORD *)(a1 + 320);
  v34 = 261;
  v9 = sub_38BF510(v8, (__int64)v33);
  if ( **(_DWORD **)(a1 + 152) != 25 )
  {
    v33[0] = "unexpected token in directive";
    v34 = 259;
    return (unsigned int)sub_3909CF0(a1, v33, 0, 0, v9, v10);
  }
  v28 = v9;
  sub_38EB180(a1);
  v27 = sub_3909290(a1 + 144);
  if ( (unsigned __int8)sub_38EB9C0(a1, &v30) )
    return 1;
  v13 = *(_DWORD **)(a1 + 152);
  v14 = v28;
  v31 = 0;
  v26 = 0;
  if ( *v13 != 25 )
    goto LABEL_10;
  sub_38EB180(a1);
  v26 = sub_3909290(a1 + 144);
  if ( (unsigned __int8)sub_38EB9C0(a1, &v31) )
    return 1;
  v17 = *(_QWORD *)(a1 + 280);
  v14 = v28;
  v18 = *(_DWORD *)(v17 + 300);
  if ( v18 )
  {
    if ( a2 )
    {
      if ( v18 != 1 )
        goto LABEL_10;
      goto LABEL_31;
    }
  }
  else if ( a2 )
  {
    v33[0] = "alignment not supported on this target";
    v34 = 259;
    return (unsigned int)sub_3909790(a1, v26, v33, 0, 0);
  }
  if ( !*(_BYTE *)(v17 + 298) )
    goto LABEL_10;
LABEL_31:
  if ( !v31 || (v31 & (v31 - 1)) != 0 )
  {
    HIBYTE(v34) = 1;
    v19 = "alignment must be a power of 2";
    goto LABEL_22;
  }
  _BitScanReverse64(&v24, v31);
  v31 = 63 - ((unsigned int)v24 ^ 0x3F);
LABEL_10:
  v29 = (__int64 *)v14;
  v33[0] = "unexpected token in '.comm' or '.lcomm' directive";
  v34 = 259;
  v11 = sub_3909E20(a1, 9, v33);
  if ( (_BYTE)v11 )
    return 1;
  v15 = v29;
  if ( v30 < 0 )
  {
    v33[0] = "invalid '.comm' or '.lcomm' directive size, can't be less than zero";
    v34 = 259;
    return (unsigned int)sub_3909790(a1, v27, v33, 0, 0);
  }
  if ( v31 < 0 )
  {
    HIBYTE(v34) = 1;
    v19 = "invalid '.comm' or '.lcomm' directive alignment, can't be less than zero";
LABEL_22:
    v33[0] = v19;
    LOBYTE(v34) = 3;
    return (unsigned int)sub_3909790(a1, v26, v33, 0, 0);
  }
  v16 = *v29;
  if ( (v29[1] & 2) != 0 )
  {
    v20 = *((_BYTE *)v29 + 9);
    if ( (v20 & 0xC) == 8 )
    {
      v20 &= 0xF3u;
      v29[3] = 0;
      *((_BYTE *)v29 + 9) = v20;
    }
    *((_BYTE *)v29 + 8) &= ~2u;
    *v29 = v16 & 7;
  }
  else
  {
    if ( (v16 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
LABEL_16:
      v33[0] = "invalid symbol redefinition";
      v34 = 259;
      return (unsigned int)sub_3909790(a1, v3, v33, 0, 0);
    }
    v20 = *((_BYTE *)v29 + 9);
  }
  if ( (v20 & 0xC) == 8 )
  {
    *((_BYTE *)v29 + 8) |= 4u;
    v25 = (unsigned __int64)sub_38CE440(v29[3]);
    v15 = v29;
    *v29 = v25 | *v29 & 7;
    if ( v25 )
      goto LABEL_16;
  }
  v21 = *(__int64 **)(a1 + 328);
  v22 = (unsigned int)(1 << v31);
  v23 = *v21;
  if ( a2 )
  {
    (*(void (__fastcall **)(__int64 *, __int64 *, __int64, __int64))(v23 + 376))(v21, v15, v30, v22);
  }
  else
  {
    (*(void (__fastcall **)(__int64 *, __int64 *, __int64, __int64))(v23 + 368))(v21, v15, v30, v22);
    return 0;
  }
  return v11;
}
