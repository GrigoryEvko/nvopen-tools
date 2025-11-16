// Function: sub_1750FB0
// Address: 0x1750fb0
//
__int64 __fastcall sub_1750FB0(_QWORD *a1, __int64 *a2, double a3, double a4, double a5)
{
  __int64 v5; // rbx
  __int64 **v6; // r14
  __int64 v7; // rax
  int v9; // ecx
  __int64 v10; // r15
  __int64 v11; // r9
  unsigned __int8 v12; // al
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 *v15; // r12
  unsigned __int8 *v16; // rax
  unsigned __int8 v17; // dl
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // r12
  __int64 *v21; // rax
  __int64 **v22; // rax
  __int64 *v23; // r10
  __int64 ****v24; // r9
  __int64 ***v25; // r9
  __int64 v26; // rdi
  __int64 v27; // rdx
  __int64 *v28; // rax
  __int64 v29; // rdi
  __int64 v30; // rdx
  unsigned __int8 *v31; // rax
  __int64 v32; // [rsp+8h] [rbp-58h]
  __int64 v33; // [rsp+8h] [rbp-58h]
  __int64 *v34; // [rsp+8h] [rbp-58h]
  __int64 v35[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v36; // [rsp+20h] [rbp-40h]

  v5 = *(a2 - 3);
  v6 = (__int64 **)*a2;
  if ( *(_BYTE *)(*(_QWORD *)v5 + 8LL) != 16 )
  {
    if ( !(unsigned __int8)sub_1705440((__int64)a1, *(_QWORD *)v5, (__int64)v6) )
      return 0;
    v5 = *(a2 - 3);
  }
  v7 = *(_QWORD *)(v5 + 8);
  if ( !v7 )
    return 0;
  if ( *(_QWORD *)(v7 + 8) )
    return 0;
  v9 = *(unsigned __int8 *)(v5 + 16);
  if ( (unsigned __int8)v9 <= 0x17u || (unsigned int)(v9 - 35) > 0x11 )
    return 0;
  if ( ((1LL << ((unsigned __int8)v9 - 24)) & 0x1C00A800) == 0 )
    return sub_17508C0(a1, a2, a3, a4, a5);
  v10 = *(_QWORD *)(v5 - 48);
  v11 = *(_QWORD *)(v5 - 24);
  v12 = *(_BYTE *)(v10 + 16);
  if ( v12 > 0x10u )
  {
    v17 = *(_BYTE *)(v11 + 16);
    if ( v17 > 0x10u )
    {
      if ( v12 > 0x17u
        && (v12 == 61 || v12 == 62)
        && ((*(_BYTE *)(v10 + 23) & 0x40) == 0
          ? (v22 = (__int64 **)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)))
          : (v22 = *(__int64 ***)(v10 - 8)),
            (v23 = *v22) != 0 && v6 == (__int64 **)*v23) )
      {
        v29 = a1[1];
        v30 = *(_QWORD *)(v5 - 24);
        v36 = 257;
        v34 = v23;
        v31 = sub_1708970(v29, 36, v30, v6, v35);
        v36 = 257;
        return sub_15FB440((unsigned int)*(unsigned __int8 *)(v5 + 16) - 24, v34, (__int64)v31, (__int64)v35, 0);
      }
      else
      {
        if ( v17 <= 0x17u || v17 != 61 && v17 != 62 )
          return sub_17508C0(a1, a2, a3, a4, a5);
        v24 = (*(_BYTE *)(v11 + 23) & 0x40) != 0
            ? *(__int64 *****)(v11 - 8)
            : (__int64 ****)(v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF));
        v25 = *v24;
        if ( !v25 || v6 != *v25 )
          return sub_17508C0(a1, a2, a3, a4, a5);
        v26 = a1[1];
        v27 = *(_QWORD *)(v5 - 48);
        v33 = (__int64)v25;
        v36 = 257;
        v28 = (__int64 *)sub_1708970(v26, 36, v27, v6, v35);
        v36 = 257;
        return sub_15FB440((unsigned int)*(unsigned __int8 *)(v5 + 16) - 24, v28, v33, (__int64)v35, 0);
      }
    }
    else
    {
      v18 = sub_15A43B0(*(_QWORD *)(v5 - 24), v6, 0);
      v19 = a1[1];
      v20 = v18;
      v36 = 257;
      v21 = (__int64 *)sub_1708970(v19, 36, v10, v6, v35);
      v36 = 257;
      return sub_15FB440((unsigned int)*(unsigned __int8 *)(v5 + 16) - 24, v21, v20, (__int64)v35, 0);
    }
  }
  else
  {
    v32 = *(_QWORD *)(v5 - 24);
    v13 = sub_15A43B0(*(_QWORD *)(v5 - 48), v6, 0);
    v14 = a1[1];
    v15 = (__int64 *)v13;
    v36 = 257;
    v16 = sub_1708970(v14, 36, v32, v6, v35);
    v36 = 257;
    return sub_15FB440((unsigned int)*(unsigned __int8 *)(v5 + 16) - 24, v15, (__int64)v16, (__int64)v35, 0);
  }
}
