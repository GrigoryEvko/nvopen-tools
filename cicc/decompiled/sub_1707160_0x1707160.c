// Function: sub_1707160
// Address: 0x1707160
//
__int64 __fastcall sub_1707160(__int64 a1, _BYTE *a2, __int64 a3, double a4, double a5, double a6)
{
  __int64 v9; // r14
  __int64 v10; // r15
  __int64 v11; // rdi
  __int64 *v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 *v16; // r14
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // r15
  _QWORD *v20; // rax
  __int64 v21; // r12
  __int64 v22; // rcx
  unsigned __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rdx
  unsigned __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  unsigned __int64 v29; // rax
  __int64 v30; // rax
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rsi
  _QWORD *v36; // [rsp+0h] [rbp-60h]
  __int64 v37; // [rsp+8h] [rbp-58h]
  __int64 v38; // [rsp+10h] [rbp-50h] BYREF
  __int16 v39; // [rsp+20h] [rbp-40h]

  v9 = *(_QWORD *)(a3 - 48);
  v10 = *(_QWORD *)(a3 - 24);
  if ( *(_BYTE *)(v9 + 16) > 0x10u && *(_BYTE *)(v10 + 16) > 0x10u )
    return 0;
  v11 = *(_QWORD *)a3;
  if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
    v11 = **(_QWORD **)(v11 + 16);
  if ( sub_1642F90(v11, 1) )
    return 0;
  if ( a2[16] == 71 )
  {
    v12 = (__int64 *)*((_QWORD *)a2 - 3);
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
    {
      v13 = *v12;
      if ( *(_BYTE *)(v13 + 8) == 16 && *(_QWORD *)(v13 + 32) == *(_QWORD *)(*(_QWORD *)a2 + 32LL) )
        goto LABEL_10;
      return 0;
    }
    if ( *(_BYTE *)(*v12 + 8) == 16 )
      return 0;
  }
LABEL_10:
  v14 = *(_QWORD *)(a3 - 72);
  if ( (unsigned __int8)(*(_BYTE *)(v14 + 16) - 75) <= 1u )
  {
    v15 = *(_QWORD *)(v14 + 8);
    if ( v15 )
    {
      if ( !*(_QWORD *)(v15 + 8) )
      {
        v32 = *(_QWORD *)(v14 - 48);
        v33 = *(_QWORD *)(a3 - 48);
        v34 = *(_QWORD *)(v14 - 24);
        v35 = *(_QWORD *)(a3 - 24);
        if ( v32 == v33 && v34 == v35 )
          return 0;
        if ( v34 == v33 && v32 == v35 )
          return 0;
      }
    }
  }
  v16 = (__int64 *)sub_1706EB0((__int64)a2, v9, *(_QWORD *)(a1 + 8), a4, a5, a6);
  v17 = sub_1706EB0((__int64)a2, v10, *(_QWORD *)(a1 + 8), a4, a5, a6);
  v18 = *(_QWORD *)(a3 - 72);
  v19 = v17;
  v39 = 257;
  v20 = sub_1648A60(56, 3u);
  v21 = (__int64)v20;
  if ( v20 )
  {
    v37 = (__int64)v20;
    v36 = v20 - 9;
    sub_15F1EA0((__int64)v20, *v16, 55, (__int64)(v20 - 9), 3, 0);
    if ( *(_QWORD *)(v21 - 72) )
    {
      v22 = *(_QWORD *)(v21 - 64);
      v23 = *(_QWORD *)(v21 - 56) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v23 = v22;
      if ( v22 )
        *(_QWORD *)(v22 + 16) = *(_QWORD *)(v22 + 16) & 3LL | v23;
    }
    *(_QWORD *)(v21 - 72) = v18;
    if ( v18 )
    {
      v24 = *(_QWORD *)(v18 + 8);
      *(_QWORD *)(v21 - 64) = v24;
      if ( v24 )
        *(_QWORD *)(v24 + 16) = (v21 - 64) | *(_QWORD *)(v24 + 16) & 3LL;
      *(_QWORD *)(v21 - 56) = (v18 + 8) | *(_QWORD *)(v21 - 56) & 3LL;
      *(_QWORD *)(v18 + 8) = v36;
    }
    if ( *(_QWORD *)(v21 - 48) )
    {
      v25 = *(_QWORD *)(v21 - 40);
      v26 = *(_QWORD *)(v21 - 32) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v26 = v25;
      if ( v25 )
        *(_QWORD *)(v25 + 16) = *(_QWORD *)(v25 + 16) & 3LL | v26;
    }
    *(_QWORD *)(v21 - 48) = v16;
    v27 = v16[1];
    *(_QWORD *)(v21 - 40) = v27;
    if ( v27 )
      *(_QWORD *)(v27 + 16) = (v21 - 40) | *(_QWORD *)(v27 + 16) & 3LL;
    *(_QWORD *)(v21 - 32) = (unsigned __int64)(v16 + 1) | *(_QWORD *)(v21 - 32) & 3LL;
    v16[1] = v21 - 48;
    if ( *(_QWORD *)(v21 - 24) )
    {
      v28 = *(_QWORD *)(v21 - 16);
      v29 = *(_QWORD *)(v21 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v29 = v28;
      if ( v28 )
        *(_QWORD *)(v28 + 16) = *(_QWORD *)(v28 + 16) & 3LL | v29;
    }
    *(_QWORD *)(v21 - 24) = v19;
    if ( v19 )
    {
      v30 = *(_QWORD *)(v19 + 8);
      *(_QWORD *)(v21 - 16) = v30;
      if ( v30 )
        *(_QWORD *)(v30 + 16) = (v21 - 16) | *(_QWORD *)(v30 + 16) & 3LL;
      *(_QWORD *)(v21 - 8) = (v19 + 8) | *(_QWORD *)(v21 - 8) & 3LL;
      *(_QWORD *)(v19 + 8) = v21 - 24;
    }
    sub_164B780(v21, &v38);
    sub_15F4370(v37, a3, 0, 0);
  }
  else
  {
    sub_15F4370(0, a3, 0, 0);
  }
  return v21;
}
