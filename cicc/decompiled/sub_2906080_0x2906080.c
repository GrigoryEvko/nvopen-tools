// Function: sub_2906080
// Address: 0x2906080
//
__int64 __fastcall sub_2906080(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 *v5; // r12
  __int64 v6; // rax
  __int64 v7; // rsi
  int v8; // edi
  __int64 v9; // r9
  __int64 v10; // rdx
  unsigned __int8 *v11; // rcx
  __int64 v12; // rcx
  __int64 v13; // r9
  __int64 v14; // r8
  __int64 v15; // rdx
  __int64 v16; // r15
  int v18; // r8d
  __int64 v19; // r10
  __int64 v20; // rax
  unsigned __int8 *v21; // r15
  int v22; // eax
  unsigned __int8 *v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  unsigned __int8 v28; // al
  __int64 v29; // rdi
  bool v30; // r15
  __int64 v31; // r12
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rax
  unsigned int v37; // eax
  __int64 *v38; // rsi
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  int v43; // eax
  int v44; // r11d
  __int64 v45; // rax
  unsigned __int8 *v46; // [rsp+8h] [rbp-38h] BYREF
  unsigned __int8 *v47; // [rsp+10h] [rbp-30h] BYREF
  __int64 v48[5]; // [rsp+18h] [rbp-28h] BYREF

  v5 = (unsigned __int8 *)a1;
  v6 = *(unsigned int *)(a2 + 24);
  v7 = *(_QWORD *)(a2 + 8);
  v46 = (unsigned __int8 *)a1;
  if ( (_DWORD)v6 )
  {
    v8 = v6 - 1;
    LODWORD(v9) = (v6 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v10 = v7 + 16LL * (unsigned int)v9;
    v11 = *(unsigned __int8 **)v10;
    if ( v5 == *(unsigned __int8 **)v10 )
    {
LABEL_3:
      v12 = *(unsigned int *)(a2 + 40);
      v13 = *(_QWORD *)(a2 + 32);
      v14 = v7 + 16 * v6;
      if ( v14 == v10 )
      {
        v15 = v13 + 16LL * *(unsigned int *)(a2 + 40);
      }
      else
      {
        v12 = v13 + 16 * v12;
        v15 = v13 + 16LL * *(unsigned int *)(v10 + 8);
        if ( v15 != v12 )
          return *(_QWORD *)(v15 + 8);
      }
      v19 = *((_QWORD *)v5 + 1);
      if ( (unsigned int)*(unsigned __int8 *)(v19 + 8) - 17 <= 1 )
      {
        v47 = v5;
        goto LABEL_12;
      }
      goto LABEL_16;
    }
    v18 = 1;
    while ( v11 != (unsigned __int8 *)-4096LL )
    {
      v9 = v8 & (unsigned int)(v18 + v9);
      v10 = v7 + 16 * v9;
      v11 = *(unsigned __int8 **)v10;
      if ( v5 == *(unsigned __int8 **)v10 )
        goto LABEL_3;
      ++v18;
    }
  }
  v19 = *((_QWORD *)v5 + 1);
  v13 = *(_QWORD *)(a2 + 32);
  v14 = v7 + 16LL * (unsigned int)v6;
  v15 = v13 + 16LL * *(unsigned int *)(a2 + 40);
  v12 = (unsigned int)*(unsigned __int8 *)(v19 + 8) - 17;
  if ( (unsigned int)v12 <= 1 )
  {
    v47 = v5;
    if ( !(_DWORD)v6 )
    {
LABEL_25:
      v28 = *v5;
      if ( *v5 == 22 )
        goto LABEL_60;
      if ( v28 > 0x15u )
      {
        if ( v28 != 61 )
        {
          if ( v28 == 91 || v28 == 92 )
          {
            v29 = a2;
LABEL_35:
            *(_QWORD *)sub_1152A40(v29, (__int64 *)&v47, v15, v12, v14, v13) = v5;
            v48[0] = (__int64)v47;
            *(_BYTE *)sub_2905EE0(a3, v48) = 0;
            return (__int64)v47;
          }
          if ( v28 != 63 )
          {
            if ( v28 != 96 && v28 != 78 )
            {
              v29 = a2;
              if ( v28 != 85 && v28 != 34 )
                goto LABEL_35;
LABEL_61:
              *(_QWORD *)sub_1152A40(v29, (__int64 *)&v47, v15, v12, v14, v13) = v5;
              v48[0] = (__int64)v47;
              *(_BYTE *)sub_2905EE0(a3, v48) = 1;
              return (__int64)v47;
            }
            goto LABEL_56;
          }
LABEL_64:
          v23 = *(unsigned __int8 **)&v5[-32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)];
          goto LABEL_21;
        }
LABEL_60:
        v29 = a2;
        goto LABEL_61;
      }
      v38 = (__int64 *)&v47;
      v16 = sub_AC9350((__int64 **)v19);
LABEL_63:
      *(_QWORD *)sub_1152A40(a2, v38, v39, v40, v41, v42) = v16;
      v48[0] = v16;
      *(_BYTE *)sub_2905EE0(a3, v48) = 1;
      return v16;
    }
    v8 = v6 - 1;
LABEL_12:
    v12 = v8 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v20 = v7 + 16 * v12;
    v21 = *(unsigned __int8 **)v20;
    if ( v5 == *(unsigned __int8 **)v20 )
    {
LABEL_13:
      if ( v20 != v14 )
      {
        v13 += 16LL * *(unsigned int *)(v20 + 8);
        if ( v13 != v15 )
          return *(_QWORD *)(v13 + 8);
      }
    }
    else
    {
      v43 = 1;
      while ( v21 != (unsigned __int8 *)-4096LL )
      {
        v44 = v43 + 1;
        v12 = v8 & (unsigned int)(v43 + v12);
        v20 = v7 + 16LL * (unsigned int)v12;
        v21 = *(unsigned __int8 **)v20;
        if ( v5 == *(unsigned __int8 **)v20 )
          goto LABEL_13;
        v43 = v44;
      }
    }
    goto LABEL_25;
  }
LABEL_16:
  v22 = *v5;
  if ( (_BYTE)v22 == 22 )
    goto LABEL_55;
  if ( (unsigned __int8)v22 <= 0x15u )
  {
    v38 = (__int64 *)&v46;
    v16 = sub_AC9EC0((__int64 **)v19);
    goto LABEL_63;
  }
  if ( (_BYTE)v22 == 77 )
    goto LABEL_55;
  v15 = (unsigned int)(v22 - 67);
  if ( (unsigned __int8)(v22 - 67) <= 0xCu )
  {
    v23 = sub_BD3990(v5, v7);
LABEL_21:
    v48[0] = (__int64)v5;
    v16 = sub_2906080(v23, a2, a3);
    *(_QWORD *)sub_1152A40(a2, v48, v24, v25, v26, v27) = v16;
    return v16;
  }
  switch ( (_BYTE)v22 )
  {
    case '=':
LABEL_55:
      *(_QWORD *)sub_1152A40(a2, (__int64 *)&v46, v15, v12, v14, v13) = v5;
      v48[0] = (__int64)v46;
      *(_BYTE *)sub_2905EE0(a3, v48) = 1;
      return (__int64)v46;
    case '?':
      goto LABEL_64;
    case '`':
LABEL_56:
      v23 = (unsigned __int8 *)*((_QWORD *)v5 - 4);
      goto LABEL_21;
    case 'U':
      v36 = *((_QWORD *)v5 - 4);
      if ( v36 && !*(_BYTE *)v36 && *(_QWORD *)(v36 + 24) == *((_QWORD *)v5 + 10) && (*(_BYTE *)(v36 + 33) & 0x20) != 0 )
      {
        v37 = *(_DWORD *)(v36 + 36);
        if ( v37 == 147 )
          goto LABEL_64;
        if ( v37 > 0x92 && (v37 == 183 || v37 <= 0xB7 && (v37 & 0xFFFFFFFD) == 0x95) )
          BUG();
      }
      goto LABEL_55;
    case '"':
    case 'A':
    case 'B':
    case ']':
      goto LABEL_55;
  }
  v30 = 0;
  if ( (unsigned __int8)v22 > 0x1Cu && (*((_QWORD *)v5 + 6) || (v5[7] & 0x20) != 0) )
  {
    v45 = sub_B91F50((__int64)v5, "is_base_value", 0xDu);
    v5 = v46;
    v30 = v45 != 0;
  }
  v48[0] = (__int64)v5;
  *(_BYTE *)sub_2905EE0(a3, v48) = v30;
  v31 = (__int64)v46;
  *(_QWORD *)sub_1152A40(a2, (__int64 *)&v46, v32, v33, v34, v35) = v31;
  return (__int64)v46;
}
