// Function: sub_8EC3E0
// Address: 0x8ec3e0
//
unsigned __int8 *__fastcall sub_8EC3E0(
        __int64 a1,
        unsigned __int64 a2,
        _DWORD *a3,
        _DWORD *a4,
        _QWORD *a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v7; // r11
  unsigned __int8 *v8; // r15
  unsigned __int64 v10; // r12
  unsigned __int8 v11; // al
  unsigned __int8 *v12; // rdi
  unsigned __int8 v13; // al
  __int64 v15; // rax
  __int64 v16; // rdx
  char *v17; // rax
  unsigned __int8 *v18; // rax
  unsigned __int8 *v19; // rax
  unsigned __int8 v20; // al
  unsigned __int8 *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // [rsp-8h] [rbp-88h]
  __int64 v28; // [rsp+8h] [rbp-78h]
  __int64 v29; // [rsp+8h] [rbp-78h]
  __int64 v31; // [rsp+20h] [rbp-60h]
  __int64 v32; // [rsp+20h] [rbp-60h]
  char *v33; // [rsp+20h] [rbp-60h]
  __int64 v34; // [rsp+20h] [rbp-60h]
  unsigned __int8 *v35; // [rsp+20h] [rbp-60h]
  _QWORD *v36; // [rsp+28h] [rbp-58h]
  int v39; // [rsp+44h] [rbp-3Ch] BYREF
  unsigned __int8 *v40; // [rsp+48h] [rbp-38h] BYREF

  v7 = 0;
  v8 = (unsigned __int8 *)a1;
  v36 = (_QWORD *)a6;
  v40 = 0;
  *a5 = 0;
  while ( 1 )
  {
    v10 = v7 + 1;
    *a3 = 0;
    *a4 = 0;
    v11 = *v8;
    if ( *v8 == 69 || !v11 )
    {
      if ( *(_DWORD *)(a7 + 24) )
      {
LABEL_19:
        if ( v11 != 73 )
          goto LABEL_20;
        goto LABEL_38;
      }
      v15 = *(_QWORD *)(a7 + 48);
      ++*(_QWORD *)(a7 + 32);
      *(_DWORD *)(a7 + 24) = 1;
      v16 = v15 + 1;
      *(_QWORD *)(a7 + 48) = v15 + 1;
      v11 = *v8;
      if ( *v8 == 73 )
        goto LABEL_26;
LABEL_20:
      if ( v11 == 69 )
        break;
LABEL_29:
      if ( !*(_QWORD *)(a7 + 48) )
        sub_8E5DC0(a1, 1, v10, 0, a7, a6);
      goto LABEL_31;
    }
    if ( v11 != 83 )
    {
      switch ( v11 )
      {
        case 'T':
          v32 = v7;
          v21 = sub_8E5C30((__int64)v8, a7);
          v7 = v32;
          v8 = v21;
          v11 = *v21;
          if ( v11 != 73 )
            goto LABEL_20;
          v16 = *(_QWORD *)(a7 + 48);
          break;
        case 'D':
          if ( (v8[1] & 0xDF) != 0x54 )
          {
            if ( v8[1] != 67 )
            {
LABEL_8:
              *a3 = 1;
              if ( *v8 != 68 )
              {
LABEL_9:
                v12 = v40;
                if ( v40 && *v40 != 83 )
                {
                  v13 = v8[1];
                  if ( (unsigned __int8)(v13 - 49) <= 1u || v13 == 57 )
                    goto LABEL_55;
                  if ( *v8 != 67 )
                  {
                    if ( v13 != 55 && v13 != 48 )
                      goto LABEL_16;
LABEL_55:
                    *a5 = v8 + 1;
                    if ( v8[1] == 73 )
                    {
                      ++*(_QWORD *)(a7 + 64);
                      v29 = v7;
                      v35 = v8 + 3;
                      v8 = (unsigned __int8 *)sub_8E9FF0((__int64)(v8 + 3), 0, 0, 0, 0, a7);
                      sub_8EB260(v35, 0, 0, a7);
                      --*(_QWORD *)(a7 + 64);
                      v7 = v29;
                    }
                    else
                    {
                      v34 = v7;
                      v8 += 2;
                      sub_8EBEA0(v12, &v39, a7);
                      v7 = v34;
                    }
                    if ( *v8 == 66 )
                      v8 = sub_8E5930(v8, a7);
                    goto LABEL_18;
                  }
                  if ( v13 == 51 || v13 == 56 || v13 == 73 && (unsigned __int8)(v8[2] - 49) <= 1u )
                    goto LABEL_55;
                }
LABEL_16:
                if ( !*(_DWORD *)(a7 + 24) )
                {
                  ++*(_QWORD *)(a7 + 32);
                  ++*(_QWORD *)(a7 + 48);
                  *(_DWORD *)(a7 + 24) = 1;
                }
                goto LABEL_18;
              }
              v22 = *(_QWORD *)(a7 + 32);
              if ( v8[1] == 55 )
              {
                if ( v22 )
                  goto LABEL_9;
                v26 = *(_QWORD *)(a7 + 8);
                v24 = v26 + 1;
                if ( *(_DWORD *)(a7 + 28) )
                  goto LABEL_65;
                v25 = *(_QWORD *)(a7 + 16);
                if ( v25 <= v24 )
                  goto LABEL_63;
                *(_BYTE *)(*(_QWORD *)a7 + v26) = 33;
                v24 = *(_QWORD *)(a7 + 8) + 1LL;
              }
              else
              {
                if ( v22 )
                  goto LABEL_9;
                v23 = *(_QWORD *)(a7 + 8);
                v24 = v23 + 1;
                if ( !*(_DWORD *)(a7 + 28) )
                {
                  v25 = *(_QWORD *)(a7 + 16);
                  if ( v25 <= v24 )
                  {
LABEL_63:
                    *(_DWORD *)(a7 + 28) = 1;
                    if ( v25 )
                    {
                      *(_BYTE *)(*(_QWORD *)a7 + v25 - 1) = 0;
                      v24 = *(_QWORD *)(a7 + 8) + 1LL;
                    }
                    goto LABEL_65;
                  }
                  *(_BYTE *)(*(_QWORD *)a7 + v23) = 126;
                  v24 = *(_QWORD *)(a7 + 8) + 1LL;
                }
              }
LABEL_65:
              *(_QWORD *)(a7 + 8) = v24;
              goto LABEL_9;
            }
LABEL_41:
            v31 = v7;
            v40 = v8;
            v18 = sub_8EBEA0(v8, a3, a7);
            v7 = v31;
            v8 = v18;
LABEL_18:
            v11 = *v8;
            if ( *v8 != 77 )
              goto LABEL_19;
            v11 = *++v8;
            if ( v11 == 73 )
            {
LABEL_38:
              v16 = *(_QWORD *)(a7 + 48);
              break;
            }
            goto LABEL_20;
          }
          v28 = v7;
          v33 = sub_8E9FF0((__int64)v8, 0, 0, 0, 1u, a7);
          sub_8EB260(v8, 0, 0, a7);
          v7 = v28;
          v11 = *v33;
          v8 = (unsigned __int8 *)v33;
          if ( *v33 != 73 )
            goto LABEL_20;
          v16 = *(_QWORD *)(a7 + 48);
          break;
        case 'C':
          goto LABEL_8;
        default:
          goto LABEL_41;
      }
LABEL_26:
      if ( !v16 )
        sub_8E5DC0(a1, 2, v7, 0, a7, a6);
LABEL_28:
      v17 = sub_8E9020(v8, a7);
      *a4 = 1;
      v8 = (unsigned __int8 *)v17;
      if ( *v17 == 69 )
        break;
      goto LABEL_29;
    }
    v19 = (unsigned __int8 *)sub_8EC8F0((_DWORD)v8, 0, 0, 0, 0, (unsigned int)&v40, 0, a7);
    a6 = v27;
    v8 = v19;
    v20 = *v19;
    if ( v20 != 69 )
    {
      if ( v20 != 73 )
        goto LABEL_31;
      goto LABEL_28;
    }
    if ( *(_DWORD *)(a7 + 24) )
      break;
    ++*(_QWORD *)(a7 + 32);
    ++*(_QWORD *)(a7 + 48);
    *(_DWORD *)(a7 + 24) = 1;
    if ( *v8 == 73 )
      goto LABEL_28;
    if ( *v8 == 69 )
      break;
LABEL_31:
    if ( *(_DWORD *)(a7 + 24) || a2 && v10 >= a2 )
      break;
    if ( !*(_QWORD *)(a7 + 32) )
      sub_8E5790((unsigned __int8 *)"::", a7);
    v7 = v10;
  }
  if ( v36 )
    *v36 = v40;
  return v8;
}
