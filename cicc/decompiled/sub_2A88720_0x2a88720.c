// Function: sub_2A88720
// Address: 0x2a88720
//
bool __fastcall sub_2A88720(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v6; // r12
  unsigned __int64 v7; // r14
  __int64 v8; // rdx
  __int64 v9; // rax
  _BYTE *v10; // r8
  __int64 v11; // rsi
  __int64 v12; // rdx
  char v13; // al
  unsigned __int64 v14; // rdx
  unsigned __int8 v15; // r9
  bool result; // al
  char v17; // di
  unsigned int v18; // eax
  unsigned __int64 v19; // rax
  unsigned __int8 v20; // cl
  __int64 v21; // rdi
  int v22; // esi
  __int64 *v23; // rax
  char v24; // r13
  _DWORD *v25; // rax
  __int64 v26; // rdi
  int v27; // esi
  __int64 *v28; // rax
  _DWORD *v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rdi
  char v32; // al
  unsigned __int64 v33; // [rsp+0h] [rbp-80h]
  _BYTE *v34; // [rsp+0h] [rbp-80h]
  _BYTE *v35; // [rsp+8h] [rbp-78h]
  unsigned __int64 v36; // [rsp+8h] [rbp-78h]
  unsigned __int8 v37; // [rsp+10h] [rbp-70h]
  _BYTE *v38; // [rsp+10h] [rbp-70h]
  unsigned __int8 v39; // [rsp+10h] [rbp-70h]
  unsigned __int8 v40; // [rsp+10h] [rbp-70h]
  unsigned __int8 v42; // [rsp+18h] [rbp-68h]
  unsigned __int8 v43; // [rsp+18h] [rbp-68h]
  unsigned __int8 v44; // [rsp+18h] [rbp-68h]
  __int64 v45; // [rsp+20h] [rbp-60h] BYREF
  __int64 v46; // [rsp+28h] [rbp-58h] BYREF
  __int64 v47; // [rsp+30h] [rbp-50h]
  __int64 v48; // [rsp+38h] [rbp-48h]
  __int64 v49; // [rsp+40h] [rbp-40h]
  __int64 v50; // [rsp+48h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 8);
  if ( v3 == a2 )
    return 1;
  v6 = sub_B2BEC0(a3);
  v47 = sub_9208B0(v6, v3);
  v7 = v47;
  v48 = v8;
  v37 = v8;
  v9 = sub_9208B0(v6, a2);
  v10 = (_BYTE *)a1;
  v11 = v9;
  v50 = v12;
  v13 = *(_BYTE *)(v3 + 8);
  v14 = v11;
  v49 = v11;
  v15 = v50;
  if ( v13 != 18 )
  {
    if ( *(_BYTE *)(a2 + 8) == 18
      || (unsigned __int8)(*(_BYTE *)(a2 + 8) - 15) <= 1u
      || (unsigned __int8)(v13 - 15) <= 1u
      || v47 != ((v47 + 7) & 0xFFFFFFFFFFFFFFF8LL) )
    {
      return 0;
    }
    v20 = v37;
    if ( v37 )
      goto LABEL_15;
    goto LABEL_11;
  }
  v17 = *(_BYTE *)(a2 + 8);
  result = v47 == v11 && v17 == 18;
  if ( result )
    return (_BYTE)v50 == v37;
  if ( v17 != 17 )
    return 0;
  if ( **(_QWORD **)(a2 + 16) == **(_QWORD **)(v3 + 16) )
  {
    v38 = (_BYTE *)a1;
    v42 = v50;
    v45 = *(_QWORD *)(a3 + 120);
    v46 = sub_A74680(&v45);
    v18 = sub_A73930(&v46);
    v15 = v42;
    v10 = v38;
    v14 = v11;
    v19 = v7 * v18;
    v7 = (v19 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v19 != v7 )
      return 0;
LABEL_11:
    v20 = v15;
    if ( v15 )
      return 0;
LABEL_15:
    if ( v14 > v7 )
      return 0;
    v21 = v3;
    v22 = *(unsigned __int8 *)(v3 + 8);
    if ( (unsigned int)(v22 - 17) <= 1 )
    {
      v23 = *(__int64 **)(v3 + 16);
      v21 = *v23;
      LOBYTE(v22) = *(_BYTE *)(*v23 + 8);
    }
    v24 = 0;
    if ( (_BYTE)v22 == 14 )
    {
      v33 = v14;
      v35 = v10;
      v39 = v20;
      v43 = v15;
      v25 = sub_AE2980(v6, *(_DWORD *)(v21 + 8) >> 8);
      v14 = v33;
      v10 = v35;
      v24 = *((_BYTE *)v25 + 16);
      v20 = v39;
      v15 = v43;
    }
    v26 = a2;
    v27 = *(unsigned __int8 *)(a2 + 8);
    if ( (unsigned int)(v27 - 17) <= 1 )
    {
      v28 = *(__int64 **)(a2 + 16);
      v26 = *v28;
      LOBYTE(v27) = *(_BYTE *)(*v28 + 8);
    }
    v36 = v14;
    v40 = v20;
    v44 = v15;
    if ( (_BYTE)v27 == 14
      && (v34 = v10, v29 = sub_AE2980(v6, *(_DWORD *)(v26 + 8) >> 8), v10 = v34, *((_BYTE *)v29 + 16)) )
    {
      if ( v24 )
      {
        v30 = v3;
        if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
          v30 = **(_QWORD **)(v3 + 16);
        v31 = a2;
        if ( (unsigned int)*(unsigned __int8 *)(a2 + 8) - 17 <= 1 )
          v31 = **(_QWORD **)(a2 + 16);
        if ( *(_DWORD *)(v31 + 8) >> 8 == *(_DWORD *)(v30 + 8) >> 8 )
        {
          v32 = sub_BCEA30(v3);
          if ( v36 == v7 && !v32 && v44 == v40 )
          {
LABEL_33:
            if ( *(_BYTE *)(v3 + 8) != 20 && *(_BYTE *)(a2 + 8) != 20 )
              return 1;
          }
        }
        return 0;
      }
    }
    else if ( !v24 )
    {
      goto LABEL_33;
    }
    if ( *v10 <= 0x15u )
      return sub_AC30F0((__int64)v10);
    return 0;
  }
  return result;
}
