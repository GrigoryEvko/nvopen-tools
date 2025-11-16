// Function: sub_13F9040
// Address: 0x13f9040
//
unsigned __int64 __fastcall sub_13F9040(
        __int64 a1,
        __int64 a2,
        unsigned __int8 a3,
        __int64 a4,
        unsigned __int64 *a5,
        int a6,
        _QWORD *a7,
        _BYTE *a8,
        _DWORD *a9)
{
  int v10; // ebx
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rcx
  unsigned __int64 *v15; // r14
  __int64 v16; // r12
  _QWORD *i; // rax
  unsigned __int64 v18; // rbx
  char v19; // al
  __int64 v20; // rax
  __int64 v21; // r15
  __int64 v22; // rcx
  unsigned __int8 v23; // al
  __int64 v24; // r13
  unsigned int v25; // eax
  __int64 v26; // rsi
  __int64 v27; // r10
  unsigned __int64 v28; // r9
  __int64 v29; // rax
  __int64 v30; // rcx
  unsigned __int64 v31; // r8
  unsigned __int8 v33; // al
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rsi
  int v37; // eax
  __int64 v38; // rax
  __int64 v40; // [rsp+20h] [rbp-90h]
  __int64 v41; // [rsp+28h] [rbp-88h]
  __int64 v42; // [rsp+30h] [rbp-80h]
  unsigned __int64 v43; // [rsp+38h] [rbp-78h]
  unsigned __int64 v44; // [rsp+38h] [rbp-78h]
  int v46; // [rsp+48h] [rbp-68h]
  __int64 v47; // [rsp+48h] [rbp-68h]
  __int64 v48; // [rsp+48h] [rbp-68h]
  __m128i v49; // [rsp+50h] [rbp-60h] BYREF
  __int64 v50; // [rsp+60h] [rbp-50h]
  __int64 v51; // [rsp+68h] [rbp-48h]
  __int64 v52; // [rsp+70h] [rbp-40h]
  char v53; // [rsp+78h] [rbp-38h]

  v10 = a6;
  v40 = a2;
  if ( !a6 )
    v10 = -1;
  v11 = sub_157EB90(a4);
  v12 = 1;
  v41 = sub_1632FA0(v11);
  while ( 2 )
  {
    switch ( *(_BYTE *)(a2 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v29 = *(_QWORD *)(a2 + 32);
        a2 = *(_QWORD *)(a2 + 24);
        v12 *= v29;
        continue;
      case 1:
        v13 = 16;
        break;
      case 2:
        v13 = 32;
        break;
      case 3:
      case 9:
        v13 = 64;
        break;
      case 4:
        v13 = 80;
        break;
      case 5:
      case 6:
        v13 = 128;
        break;
      case 7:
        v13 = 8 * (unsigned int)sub_15A9520(v41, 0);
        break;
      case 0xB:
        v13 = *(_DWORD *)(a2 + 8) >> 8;
        break;
      case 0xD:
        v13 = 8LL * *(_QWORD *)sub_15A9930(v41, a2);
        break;
      case 0xE:
        v24 = *(_QWORD *)(a2 + 32);
        v47 = *(_QWORD *)(a2 + 24);
        v25 = sub_15A9FE0(v41, v47);
        v26 = v47;
        v27 = 1;
        v28 = v25;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v26 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v38 = *(_QWORD *)(v26 + 32);
              v26 = *(_QWORD *)(v26 + 24);
              v27 *= v38;
              continue;
            case 1:
              v35 = 16;
              goto LABEL_58;
            case 2:
              v35 = 32;
              goto LABEL_58;
            case 3:
            case 9:
              v35 = 64;
              goto LABEL_58;
            case 4:
              v35 = 80;
              goto LABEL_58;
            case 5:
            case 6:
              v35 = 128;
              goto LABEL_58;
            case 7:
              v44 = v28;
              v36 = 0;
              v48 = v27;
              goto LABEL_61;
            case 0xB:
              v35 = *(_DWORD *)(v26 + 8) >> 8;
              goto LABEL_58;
            case 0xD:
              JUMPOUT(0x13F9543);
            case 0xE:
              v42 = *(_QWORD *)(v26 + 24);
              sub_15A9FE0(v41, v42);
              sub_127FA20(v41, v42);
              JUMPOUT(0x13F951E);
            case 0xF:
              v44 = v28;
              v48 = v27;
              v36 = *(_DWORD *)(v26 + 8) >> 8;
LABEL_61:
              v37 = sub_15A9520(v41, v36);
              v27 = v48;
              v28 = v44;
              v35 = (unsigned int)(8 * v37);
LABEL_58:
              v13 = 8 * v28 * v24 * ((v28 + ((unsigned __int64)(v35 * v27 + 7) >> 3) - 1) / v28);
              break;
          }
          break;
        }
        break;
      case 0xF:
        v13 = 8 * (unsigned int)sub_15A9520(v41, *(_DWORD *)(a2 + 8) >> 8);
        break;
    }
    break;
  }
  v14 = v12;
  v15 = a5;
  v43 = (unsigned __int64)(v13 * v14 + 7) >> 3;
  v46 = v10;
  v16 = sub_1649C60(a1);
  for ( i = (_QWORD *)*a5; ; i = (_QWORD *)*v15 )
  {
    do
    {
      if ( *(_QWORD **)(a4 + 48) == i )
        return 0;
      v18 = *i & 0xFFFFFFFFFFFFFFF8LL;
      *v15 = v18;
      i = (_QWORD *)v18;
      if ( !v18 )
        BUG();
      if ( *(_BYTE *)(v18 - 8) != 78 )
        break;
      v34 = *(_QWORD *)(v18 - 48);
      if ( *(_BYTE *)(v34 + 16) )
        break;
    }
    while ( (*(_BYTE *)(v34 + 33) & 0x20) != 0 && (unsigned int)(*(_DWORD *)(v34 + 36) - 35) <= 3 );
    *v15 = *(_QWORD *)(v18 + 8);
    if ( a9 )
      ++*a9;
    if ( !v46 )
      return 0;
    *v15 = *(_QWORD *)*v15 & 0xFFFFFFFFFFFFFFF8LL;
    v19 = *(_BYTE *)(v18 - 8);
    if ( v19 == 54 )
    {
      v20 = sub_1649C60(*(_QWORD *)(v18 - 48));
      if ( (unsigned __int8)sub_13F74E0(v20, v16) && (unsigned __int8)sub_15FBDD0(*(_QWORD *)(v18 - 24), v40, v41) )
      {
        if ( (unsigned __int8)sub_15F32D0(v18 - 24) >= a3 )
        {
          v31 = v18 - 24;
          if ( a8 )
            *a8 = 1;
          return v31;
        }
        return 0;
      }
      v19 = *(_BYTE *)(v18 - 8);
    }
    if ( v19 == 55 )
      break;
    if ( (unsigned __int8)sub_15F3040(v18 - 24) )
    {
      if ( !a7
        || (v53 = 1,
            v49.m128i_i64[0] = v16,
            v49.m128i_i64[1] = v43,
            v50 = 0,
            v51 = 0,
            v52 = 0,
            (sub_13575E0(a7, v18 - 24, &v49, v30) & 2) != 0) )
      {
LABEL_39:
        *v15 = *(_QWORD *)(*v15 + 8);
        return 0;
      }
    }
LABEL_23:
    --v46;
  }
  v21 = sub_1649C60(*(_QWORD *)(v18 - 48));
  if ( !(unsigned __int8)sub_13F74E0(v21, v16) || !(unsigned __int8)sub_15FBDD0(**(_QWORD **)(v18 - 72), v40, v41) )
  {
    v23 = *(_BYTE *)(v16 + 16);
    if ( v23 <= 0x17u )
    {
      if ( v23 != 3 )
        goto LABEL_21;
    }
    else if ( v23 != 53 )
    {
LABEL_21:
      if ( !a7 )
        goto LABEL_39;
      v49.m128i_i64[0] = v16;
      v50 = 0;
      v49.m128i_i64[1] = v43;
      v51 = 0;
      v52 = 0;
      if ( (sub_134D0E0((__int64)a7, v18 - 24, &v49, v22) & 2) != 0 )
        goto LABEL_39;
      goto LABEL_23;
    }
    v33 = *(_BYTE *)(v21 + 16);
    if ( v33 <= 0x17u )
    {
      if ( v33 != 3 )
        goto LABEL_21;
    }
    else if ( v33 != 53 )
    {
      goto LABEL_21;
    }
    if ( v16 != v21 )
      goto LABEL_23;
    goto LABEL_21;
  }
  if ( (unsigned __int8)sub_15F32D0(v18 - 24) < a3 )
    return 0;
  if ( a8 )
    *a8 = 0;
  return *(_QWORD *)(v18 - 72);
}
