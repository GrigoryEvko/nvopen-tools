// Function: sub_3872990
// Address: 0x3872990
//
__int64 __fastcall sub_3872990(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int8 a6)
{
  __int64 v6; // r10
  __int64 v7; // r13
  __int64 v8; // r12
  unsigned __int8 v9; // bl
  unsigned int v10; // r11d
  unsigned __int16 v12; // ax
  unsigned int v13; // edx
  __int64 v14; // r10
  unsigned __int16 v15; // ax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rbx
  __int64 v20; // rax
  unsigned int v21; // esi
  unsigned __int8 *v22; // rax
  unsigned __int8 *v23; // rcx
  unsigned int v24; // esi
  int v25; // eax
  __int64 v26; // rax
  __int64 v27; // r10
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  _QWORD *v31; // rax
  unsigned int v32; // r15d
  _QWORD *v33; // rbx
  char v34; // al
  unsigned __int64 v35; // rax
  __int64 v36; // [rsp+8h] [rbp-68h]
  _QWORD *v37; // [rsp+8h] [rbp-68h]
  __int64 v38; // [rsp+10h] [rbp-60h]
  __int64 v39; // [rsp+10h] [rbp-60h]
  unsigned __int8 v40; // [rsp+10h] [rbp-60h]
  __int64 v41; // [rsp+10h] [rbp-60h]
  __int64 v42; // [rsp+18h] [rbp-58h]
  __int64 v43; // [rsp+18h] [rbp-58h]
  unsigned __int8 v44; // [rsp+18h] [rbp-58h]
  char v45; // [rsp+18h] [rbp-58h]
  unsigned __int8 v46; // [rsp+18h] [rbp-58h]
  __int64 v47; // [rsp+18h] [rbp-58h]
  unsigned __int8 v48; // [rsp+18h] [rbp-58h]
  _BYTE v49[16]; // [rsp+20h] [rbp-50h] BYREF
  char v50; // [rsp+30h] [rbp-40h]

  while ( 1 )
  {
    v6 = a5;
    v7 = a3;
    v8 = a4;
    v9 = a6;
    if ( a4 )
    {
      v42 = a5;
      sub_38727B0((__int64)v49, a1, a2, a4, a3);
      v6 = v42;
      if ( v50 )
        return 0;
    }
    v12 = *(_WORD *)(a2 + 24);
    if ( v12 > 3u )
      break;
    if ( !v12 )
      return 0;
    a2 = *(_QWORD *)(a2 + 32);
    a6 = v9;
    a4 = v8;
    a3 = v7;
    a5 = v6;
  }
  v10 = 0;
  if ( v12 == 10 )
    return v10;
  v43 = v6;
  sub_1412190(v6, a2);
  v14 = v43;
  v10 = v13;
  if ( !(_BYTE)v13 )
    return 0;
  v15 = *(_WORD *)(a2 + 24);
  if ( v15 != 6 )
    goto LABEL_25;
  v16 = *(_QWORD *)(a2 + 40);
  if ( !*(_WORD *)(v16 + 24) )
  {
    v17 = *(_QWORD *)(v16 + 32);
    if ( *(_DWORD *)(v17 + 32) > 0x40u )
    {
      v38 = v43;
      v45 = v13;
      v25 = sub_16A5940(v17 + 24);
      LOBYTE(v10) = v45;
      v14 = v38;
      if ( v25 == 1 )
      {
LABEL_15:
        v44 = v10;
        v19 = sub_1632FA0(*(_QWORD *)(*(_QWORD *)(**(_QWORD **)(v7 + 32) + 56LL) + 40LL));
        v20 = sub_1456040(*(_QWORD *)(a2 + 40));
        v10 = v44;
        v21 = *(_DWORD *)(v20 + 8);
        v22 = *(unsigned __int8 **)(v19 + 24);
        v23 = &v22[*(unsigned int *)(v19 + 32)];
        v24 = v21 >> 8;
        if ( v23 == v22 )
          return v10;
        while ( *v22 != v24 )
        {
          if ( v23 == ++v22 )
            return v10;
        }
        return 0;
      }
    }
    else
    {
      v18 = *(_QWORD *)(v17 + 24);
      if ( v18 && (v18 & (v18 - 1)) == 0 )
        goto LABEL_15;
    }
  }
  v46 = v10;
  v39 = v14;
  v26 = sub_13F9E70(v7);
  v10 = v46;
  if ( v26 )
  {
    v27 = v39;
    if ( !v8 )
    {
      v35 = *(_QWORD *)(v26 + 40) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v35 )
        v8 = v35 - 24;
    }
    v40 = v46;
    v36 = v27;
    v47 = *a1;
    v28 = sub_1456040(a2);
    v29 = sub_145CF80(v47, v28, 1, 0);
    v30 = sub_13A5B00(v47, a2, v29, 0, 0);
    sub_38727B0((__int64)v49, a1, v30, v8, v7);
    v10 = v40;
    if ( v50 )
    {
      v15 = *(_WORD *)(a2 + 24);
      v14 = v36;
LABEL_25:
      if ( !v9 || (unsigned __int16)(v15 - 8) > 1u )
      {
        if ( (unsigned int)v15 - 4 > 1 && (unsigned __int16)(v15 - 7) > 2u )
          return 0;
        v31 = *(_QWORD **)(a2 + 32);
        v37 = &v31[*(_QWORD *)(a2 + 40)];
        if ( v31 == v37 )
          return 0;
        v32 = v9;
        v33 = *(_QWORD **)(a2 + 32);
        while ( 1 )
        {
          v48 = v10;
          v41 = v14;
          v34 = sub_3872990(a1, *v33, v7, v8, v14, v32);
          v10 = v48;
          if ( v34 )
            break;
          ++v33;
          v14 = v41;
          if ( v37 == v33 )
            return 0;
        }
      }
    }
  }
  return v10;
}
