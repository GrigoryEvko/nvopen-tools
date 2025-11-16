// Function: sub_DDBDC0
// Address: 0xddbdc0
//
__int64 __fastcall sub_DDBDC0(__int64 a1, unsigned __int64 a2, __int64 a3, _BYTE *a4, __int64 a5, char a6, __int64 a7)
{
  __int64 *v11; // rax
  __int64 v12; // rax
  __int64 *v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r9
  unsigned int v16; // r8d
  __int64 *v17; // rax
  char v19; // al
  unsigned int v20; // edx
  __int64 v21; // rdi
  bool v22; // zf
  unsigned __int8 v23; // al
  __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // rcx
  unsigned __int8 v27; // al
  __int64 v28; // r8
  _QWORD *v29; // rsi
  _QWORD *v30; // rdx
  _QWORD *v31; // rax
  __int64 v32; // rcx
  __int64 *v33; // rdx
  __int64 v34; // r8
  char v35; // bl
  __int64 *v36; // rbx
  __int64 *v37; // rax
  unsigned int v38; // eax
  __int64 *v39; // rax
  __int64 v40; // r8
  unsigned int v41; // eax
  __int64 *v42; // rax
  _BYTE *v43; // rdi
  __int64 v44; // [rsp+8h] [rbp-58h]
  __int64 v45; // [rsp+8h] [rbp-58h]
  __int64 v46; // [rsp+10h] [rbp-50h]
  unsigned __int64 v47; // [rsp+10h] [rbp-50h]
  __int64 v48; // [rsp+18h] [rbp-48h]
  int v49; // [rsp+18h] [rbp-48h]
  unsigned __int8 v51; // [rsp+28h] [rbp-38h]

  v11 = (__int64 *)sub_BD5C60(a5);
  v12 = sub_ACD760(v11, a6);
  v16 = 1;
  if ( a5 == v12 )
    return v16;
  if ( !*(_BYTE *)(a1 + 252) )
    goto LABEL_10;
  v17 = *(__int64 **)(a1 + 232);
  v14 = *(unsigned int *)(a1 + 244);
  v13 = &v17[v14];
  if ( v17 == v13 )
  {
LABEL_9:
    if ( (unsigned int)v14 < *(_DWORD *)(a1 + 240) )
    {
      *(_DWORD *)(a1 + 244) = v14 + 1;
      *v13 = a5;
      v19 = *(_BYTE *)(a1 + 252);
      ++*(_QWORD *)(a1 + 224);
      goto LABEL_11;
    }
LABEL_10:
    sub_C8CC70(a1 + 224, a5, (__int64)v13, v14, 1, v15);
    v19 = *(_BYTE *)(a1 + 252);
    v16 = v20;
    if ( !(_BYTE)v20 )
      return v16;
LABEL_11:
    if ( *(_BYTE *)a5 <= 0x1Cu )
    {
LABEL_25:
      v16 = 0;
      goto LABEL_26;
    }
    v21 = *(_QWORD *)(a5 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v21 + 8) - 17 <= 1 )
      v21 = **(_QWORD **)(v21 + 16);
    v22 = !sub_BCAC40(v21, 1);
    v23 = *(_BYTE *)a5;
    if ( !v22 )
    {
      if ( v23 == 57 )
      {
        if ( (*(_BYTE *)(a5 + 7) & 0x40) != 0 )
          v33 = *(__int64 **)(a5 - 8);
        else
          v33 = (__int64 *)(a5 - 32LL * (*(_DWORD *)(a5 + 4) & 0x7FFFFFF));
        v34 = *v33;
        if ( *v33 )
        {
          v44 = v33[4];
          if ( v44 )
          {
            if ( !a6 )
            {
LABEL_38:
              v16 = sub_DDBDC0(a1, a2, a3, (_DWORD)a4, v34, 0, a7);
              if ( (_BYTE)v16 )
              {
LABEL_39:
                v19 = *(_BYTE *)(a1 + 252);
                goto LABEL_26;
              }
              v41 = sub_DDBDC0(a1, a2, a3, (_DWORD)a4, v44, 0, a7);
LABEL_56:
              v16 = v41;
              v19 = *(_BYTE *)(a1 + 252);
LABEL_26:
              if ( v19 )
              {
                v29 = *(_QWORD **)(a1 + 232);
                v30 = &v29[*(unsigned int *)(a1 + 244)];
                v31 = v29;
                if ( v29 != v30 )
                {
                  while ( a5 != *v31 )
                  {
                    if ( v30 == ++v31 )
                      return v16;
                  }
                  v32 = (unsigned int)(*(_DWORD *)(a1 + 244) - 1);
                  *(_DWORD *)(a1 + 244) = v32;
                  *v31 = v29[v32];
                  ++*(_QWORD *)(a1 + 224);
                }
              }
              else
              {
                v51 = v16;
                v42 = sub_C8CA60(a1 + 224, a5);
                v16 = v51;
                if ( v42 )
                {
                  *v42 = -2;
                  ++*(_DWORD *)(a1 + 248);
                  ++*(_QWORD *)(a1 + 224);
                }
              }
              return v16;
            }
            goto LABEL_24;
          }
        }
        goto LABEL_42;
      }
      if ( v23 == 86 )
      {
        v24 = *(_QWORD *)(a5 + 8);
        v48 = *(_QWORD *)(a5 - 96);
        if ( *(_QWORD *)(v48 + 8) != v24 || **(_BYTE **)(a5 - 32) > 0x15u )
        {
LABEL_18:
          if ( (unsigned int)*(unsigned __int8 *)(v24 + 8) - 17 <= 1 )
            v24 = **(_QWORD **)(v24 + 16);
          v22 = !sub_BCAC40(v24, 1);
          v27 = *(_BYTE *)a5;
          if ( !v22 )
          {
            if ( v27 == 58 )
            {
              v39 = (__int64 *)sub_986520(a5);
              v40 = *v39;
              if ( !*v39 )
                goto LABEL_24;
              v45 = v39[4];
              if ( !v45 || !a6 )
                goto LABEL_24;
              goto LABEL_54;
            }
            if ( v27 == 86 )
            {
              v28 = *(_QWORD *)(a5 - 96);
              v49 = v28;
              if ( *(_QWORD *)(v28 + 8) != *(_QWORD *)(a5 + 8) )
                goto LABEL_24;
              v43 = *(_BYTE **)(a5 - 64);
              if ( *v43 > 0x15u )
                goto LABEL_24;
              v45 = *(_QWORD *)(a5 - 32);
              if ( sub_AD7A80(v43, 1, v25, v26, v28) && v45 )
              {
                LODWORD(v40) = v49;
                if ( !a6 )
                {
                  if ( *(_BYTE *)a5 == 82 )
                    goto LABEL_65;
                  goto LABEL_24;
                }
LABEL_54:
                v16 = sub_DDBDC0(a1, a2, a3, (_DWORD)a4, v40, 1, a7);
                if ( (_BYTE)v16 )
                  goto LABEL_39;
                v41 = sub_DDBDC0(a1, a2, a3, (_DWORD)a4, v45, 1, a7);
                goto LABEL_56;
              }
              v27 = *(_BYTE *)a5;
            }
          }
          if ( v27 == 82 )
          {
            v46 = 42;
            if ( a6 )
              goto LABEL_49;
LABEL_65:
            v47 = ((unsigned __int64)((*(_BYTE *)(a5 + 1) & 2) != 0) << 32)
                | *(_WORD *)(a5 + 2) & 0x3F
                | v46 & 0xFFFFFF0000000000LL;
            goto LABEL_50;
          }
LABEL_24:
          v19 = *(_BYTE *)(a1 + 252);
          goto LABEL_25;
        }
        v44 = *(_QWORD *)(a5 - 64);
        if ( sub_AC30F0(*(_QWORD *)(a5 - 32)) && v44 )
        {
          LODWORD(v34) = v48;
          if ( a6 )
          {
            if ( *(_BYTE *)a5 == 82 )
            {
LABEL_49:
              v35 = *(_BYTE *)(a5 + 1) >> 1;
              v47 = ((unsigned __int64)(v35 & 1) << 32)
                  | (unsigned int)sub_B52870(*(_WORD *)(a5 + 2) & 0x3F)
                  | v46 & 0xFFFFFF0000000000LL;
LABEL_50:
              v36 = sub_DD8400(a1, *(_QWORD *)(a5 - 64));
              v37 = sub_DD8400(a1, *(_QWORD *)(a5 - 32));
              LOBYTE(v38) = sub_DDB9F0((__int64 *)a1, a2, a3, a4, v47, v36, v37, a7);
              v16 = v38;
              v19 = *(_BYTE *)(a1 + 252);
              goto LABEL_26;
            }
            goto LABEL_24;
          }
          goto LABEL_38;
        }
        v23 = *(_BYTE *)a5;
      }
    }
    if ( v23 <= 0x1Cu )
      goto LABEL_24;
LABEL_42:
    v24 = *(_QWORD *)(a5 + 8);
    goto LABEL_18;
  }
  while ( a5 != *v17 )
  {
    if ( v13 == ++v17 )
      goto LABEL_9;
  }
  return 0;
}
