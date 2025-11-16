// Function: sub_39A8AE0
// Address: 0x39a8ae0
//
__int64 __fastcall sub_39A8AE0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rdi
  __int64 v8; // rax
  size_t v9; // rdx
  int v10; // r15d
  __int64 result; // rax
  __int64 v12; // r8
  int v13; // r8d
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 *v16; // r12
  __int64 v17; // r15
  char v18; // al
  __int64 v19; // rsi
  __int64 v20; // rax
  int v21; // eax
  __int64 v22; // r8
  __int64 v23; // rbx
  void *v24; // rcx
  size_t v25; // rdx
  size_t v26; // r8
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rdi
  void *v30; // rax
  size_t v31; // rdx
  __int64 v32; // rdi
  void *v33; // rax
  size_t v34; // rdx
  __int64 v35; // r8
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rbx
  __int64 v39; // rax
  __int64 v40; // r8
  bool v41; // r11
  __int64 v42; // r8
  unsigned int v43; // eax
  __int64 v44; // r8
  __int64 v45; // r8
  __int64 v46; // [rsp+0h] [rbp-80h]
  __int64 v47; // [rsp+8h] [rbp-78h]
  unsigned __int16 v48; // [rsp+16h] [rbp-6Ah]
  void *v49; // [rsp+18h] [rbp-68h]
  __int64 v50; // [rsp+20h] [rbp-60h]
  size_t v51; // [rsp+28h] [rbp-58h]
  size_t v52; // [rsp+30h] [rbp-50h]
  __int64 v53; // [rsp+38h] [rbp-48h]
  _DWORD v54[13]; // [rsp+4Ch] [rbp-34h] BYREF

  v5 = a3;
  v6 = *(unsigned int *)(a3 + 8);
  v7 = *(_QWORD *)(v5 + 8 * (2 - v6));
  if ( v7 )
  {
    v8 = sub_161E970(v7);
    v10 = *(unsigned __int16 *)(a2 + 28);
    v49 = (void *)v8;
    v52 = v9;
    result = *(_QWORD *)(v5 + 32) >> 3;
    v51 = v9;
    v53 = result;
    if ( (unsigned __int16)v10 <= 0x17u )
    {
      if ( (_WORD)v10 )
      {
        switch ( (__int16)v10 )
        {
          case 1:
            result = sub_39A7890(a1, a2, v5);
            if ( v52 )
              goto LABEL_23;
            break;
          case 2:
          case 19:
          case 23:
            v6 = *(unsigned int *)(v5 + 8);
            goto LABEL_33;
          case 4:
            sub_39A7AD0(a1, a2, v5);
            if ( v52 )
              goto LABEL_23;
            goto LABEL_8;
          default:
            goto LABEL_6;
        }
        return result;
      }
      if ( !v9 )
        goto LABEL_20;
LABEL_23:
      sub_39A3F30(a1, a2, 3, v49, v51);
      result = (unsigned int)(v10 - 2);
      if ( (((_WORD)v10 - 2) & 0xFFFD) == 0 )
        goto LABEL_8;
      goto LABEL_20;
    }
    if ( (_WORD)v10 != 51 )
      goto LABEL_6;
    v6 = *(unsigned int *)(v5 + 8);
LABEL_28:
    v46 = *(_QWORD *)(v5 + 8 * (8 - v6));
    v10 = 51;
    if ( v46 )
    {
      v14 = sub_39A7C50(a1, a2, v46);
      sub_39A3B20((__int64)a1, a2, 21, v14);
      v6 = *(unsigned int *)(v5 + 8);
    }
LABEL_34:
    v50 = *(_QWORD *)(v5 + 8 * (4 - v6));
    if ( v50 )
    {
      v15 = 8LL * *(unsigned int *)(v50 + 8);
      if ( v50 != v50 - v15 )
      {
        v48 = v10;
        v47 = v5;
        v16 = (__int64 *)(v50 - v15);
        do
        {
          v17 = *v16;
          if ( *v16 )
          {
            v18 = *(_BYTE *)v17;
            if ( *(_BYTE *)v17 == 17 )
            {
              sub_39A8220((__int64)a1, *v16, 0);
            }
            else if ( v18 == 12 )
            {
              if ( *(_WORD *)(v17 + 2) == 42 )
              {
                v37 = sub_39A5A90((__int64)a1, 42, a2, 0);
                sub_39A6760(a1, v37, *(_QWORD *)(v17 + 8 * (3LL - *(unsigned int *)(v17 + 8))), 65);
              }
              else if ( (*(_BYTE *)(v17 + 29) & 0x10) != 0 )
              {
                sub_39A8670(a1, *v16);
              }
              else if ( v48 == 51 )
              {
                v38 = sub_39A5A90((__int64)a1, 25, a2, 0);
                v39 = *(_QWORD *)(v17 + 8 * (4LL - *(unsigned int *)(v17 + 8)));
                if ( v39 )
                {
                  v40 = *(_QWORD *)(v39 + 136);
                  if ( v40 )
                  {
                    if ( *(_BYTE *)(v40 + 16) == 13 )
                    {
                      v41 = sub_39A1AB0(*(_QWORD *)(v46 + 8 * (3LL - *(unsigned int *)(v46 + 8))));
                      v43 = *(_DWORD *)(v42 + 32);
                      v44 = *(_QWORD *)(v42 + 24);
                      if ( v41 )
                      {
                        if ( v43 > 0x40 )
                          v44 = *(_QWORD *)v44;
                        BYTE2(v54[0]) = 0;
                        sub_39A3560((__int64)a1, (__int64 *)(v38 + 8), 22, (__int64)v54, v44);
                      }
                      else
                      {
                        if ( v43 > 0x40 )
                          v45 = *(_QWORD *)v44;
                        else
                          v45 = v44 << (64 - (unsigned __int8)v43) >> (64 - (unsigned __int8)v43);
                        BYTE2(v54[0]) = 0;
                        sub_39A3860((__int64)a1, (__int64 *)(v38 + 8), 22, (__int64)v54, v45);
                      }
                    }
                  }
                }
                sub_39A7C50(a1, v38, v17);
              }
              else
              {
                sub_39A7C50(a1, a2, *v16);
              }
            }
            else if ( v18 == 27 )
            {
              v23 = sub_39A5A90((__int64)a1, *(_WORD *)(v17 + 2), a2, 0);
              v24 = *(void **)(v17 - 8LL * *(unsigned int *)(v17 + 8));
              if ( v24 )
              {
                v24 = (void *)sub_161E970(*(_QWORD *)(v17 - 8LL * *(unsigned int *)(v17 + 8)));
                v26 = v25;
              }
              else
              {
                v26 = 0;
              }
              sub_39A3F30(a1, v23, 16360, v24, v26);
              v27 = *(_QWORD *)(v17 + 8 * (4LL - *(unsigned int *)(v17 + 8)));
              if ( v27 )
                sub_39A6760(a1, v23, v27, 73);
              sub_39A37B0((__int64)a1, v23, v17);
              v28 = *(unsigned int *)(v17 + 8);
              v29 = *(_QWORD *)(v17 + 8 * (2 - v28));
              if ( v29 )
              {
                v30 = (void *)sub_161E970(v29);
                if ( v31 )
                  sub_39A3F30(a1, v23, 16361, v30, v31);
                v28 = *(unsigned int *)(v17 + 8);
              }
              v32 = *(_QWORD *)(v17 + 8 * (3 - v28));
              if ( v32 )
              {
                v33 = (void *)sub_161E970(v32);
                if ( v34 )
                  sub_39A3F30(a1, v23, 16362, v33, v34);
              }
              v35 = *(unsigned int *)(v17 + 28);
              if ( (_DWORD)v35 )
              {
                BYTE2(v54[0]) = 0;
                sub_39A3560((__int64)a1, (__int64 *)(v23 + 8), 16363, (__int64)v54, v35);
              }
            }
            else if ( v18 == 13 && *(_WORD *)(v17 + 2) == 51 )
            {
              v36 = sub_39A5A90((__int64)a1, 51, a2, 0);
              sub_39A8AE0(a1, v36, v17);
            }
          }
          ++v16;
        }
        while ( (__int64 *)v50 != v16 );
        v10 = v48;
        v5 = v47;
      }
    }
    if ( (*(_BYTE *)(v5 + 28) & 8) != 0 )
      sub_39A34D0((__int64)a1, a2, 16356);
    v19 = *(_QWORD *)(v5 + 8 * (5LL - *(unsigned int *)(v5 + 8)));
    if ( v19 )
    {
      v20 = sub_39A64F0(a1, v19);
      sub_39A3B20((__int64)a1, a2, 29, v20);
    }
    if ( (*(_BYTE *)(v5 + 29) & 2) != 0 )
      sub_39A34D0((__int64)a1, a2, 16364);
    if ( (v10 & 0xFFFB) == 0x13 || (_WORD)v10 == 2 )
      sub_39A6D90(a1, a2, *(_QWORD *)(v5 + 8 * (6LL - *(unsigned int *)(v5 + 8))));
    v21 = *(_DWORD *)(v5 + 28);
    if ( (v21 & 0x400000) != 0 )
    {
      v22 = 5;
    }
    else
    {
      v22 = 4;
      if ( (v21 & 0x800000) == 0 )
        goto LABEL_6;
    }
    v54[0] = 65547;
    sub_39A3560((__int64)a1, (__int64 *)(a2 + 8), 54, (__int64)v54, v22);
LABEL_6:
    if ( v52 )
      goto LABEL_23;
LABEL_7:
    result = (unsigned int)(v10 - 2);
    if ( (((_WORD)v10 - 2) & 0xFFFD) == 0 )
      goto LABEL_8;
    goto LABEL_20;
  }
  v10 = *(unsigned __int16 *)(a2 + 28);
  v49 = 0;
  v51 = 0;
  result = *(_QWORD *)(v5 + 32) >> 3;
  v53 = result;
  if ( (unsigned __int16)v10 > 0x17u )
  {
    if ( (_WORD)v10 != 51 )
      goto LABEL_7;
    v52 = 0;
    goto LABEL_28;
  }
  if ( !(_WORD)v10 )
  {
LABEL_20:
    if ( (v10 & 0xFFFB) != 0x13 )
      return result;
LABEL_8:
    if ( v53 )
    {
      BYTE2(v54[0]) = 0;
      sub_39A3560((__int64)a1, (__int64 *)(a2 + 8), 11, (__int64)v54, v53);
    }
    else
    {
      if ( (*(_BYTE *)(v5 + 28) & 4) != 0 )
      {
LABEL_10:
        sub_39A34D0((__int64)a1, a2, 60);
        if ( (*(_BYTE *)(v5 + 28) & 4) != 0 )
        {
          v12 = *(unsigned int *)(v5 + 52);
          if ( !(_DWORD)v12 )
            goto LABEL_12;
LABEL_18:
          v54[0] = 65547;
          sub_39A3560((__int64)a1, (__int64 *)(a2 + 8), 16358, (__int64)v54, v12);
LABEL_12:
          result = sub_398C0A0(a1[25]);
          if ( (unsigned __int16)result > 4u )
          {
            v13 = *(_DWORD *)(v5 + 48) >> 3;
            if ( v13 )
            {
              v54[0] = 65551;
              return sub_39A3560((__int64)a1, (__int64 *)(a2 + 8), 136, (__int64)v54, v13 & 0x1FFFFFFF);
            }
          }
          return result;
        }
LABEL_17:
        sub_39A3790((__int64)a1, a2, v5);
        v12 = *(unsigned int *)(v5 + 52);
        if ( !(_DWORD)v12 )
          goto LABEL_12;
        goto LABEL_18;
      }
      BYTE2(v54[0]) = 0;
      sub_39A3560((__int64)a1, (__int64 *)(a2 + 8), 11, (__int64)v54, 0);
    }
    if ( (*(_BYTE *)(v5 + 28) & 4) == 0 )
      goto LABEL_17;
    goto LABEL_10;
  }
  switch ( (__int16)v10 )
  {
    case 1:
      result = sub_39A7890(a1, a2, v5);
      break;
    case 2:
    case 19:
    case 23:
      v52 = 0;
LABEL_33:
      v46 = 0;
      goto LABEL_34;
    case 4:
      sub_39A7AD0(a1, a2, v5);
      goto LABEL_8;
    default:
      goto LABEL_7;
  }
  return result;
}
