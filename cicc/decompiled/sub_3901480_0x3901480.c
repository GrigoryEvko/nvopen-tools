// Function: sub_3901480
// Address: 0x3901480
//
__int64 __fastcall sub_3901480(__int64 a1)
{
  __int64 v2; // r8
  __int64 v3; // r9
  const char *v4; // rax
  __int64 v5; // rdi
  __int64 result; // rax
  __int64 v7; // rax
  unsigned __int64 v8; // r13
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // r14
  unsigned __int64 v11; // r13
  int v12; // r15d
  __int64 v13; // rdi
  unsigned __int8 v14; // bl
  int v15; // eax
  __int64 v16; // rax
  unsigned __int64 v17; // rsi
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // r15
  __int64 v22; // r9
  _BYTE *v23; // rsi
  _BYTE *v24; // rdx
  __int64 v25; // r8
  unsigned int v26; // ecx
  const char *v27; // rax
  __int64 v28; // rdi
  unsigned int v29; // eax
  unsigned int v30; // r9d
  int v31; // eax
  bool v32; // zf
  int v33; // r9d
  int v34; // r8d
  int v35; // eax
  int v36; // edi
  int v37; // esi
  int v38; // edx
  int v39; // ecx
  int v40; // eax
  unsigned int v41; // eax
  unsigned int v42; // r9d
  int v43; // eax
  unsigned int v44; // r8d
  unsigned int v45; // eax
  __int64 v46; // rdi
  char v47; // al
  unsigned int v48; // [rsp+1Ch] [rbp-64h] BYREF
  __int64 v49; // [rsp+20h] [rbp-60h] BYREF
  __int64 v50; // [rsp+28h] [rbp-58h]
  _QWORD v51[2]; // [rsp+30h] [rbp-50h] BYREF
  char v52; // [rsp+40h] [rbp-40h]
  char v53; // [rsp+41h] [rbp-3Fh]

  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 2 )
    goto LABEL_2;
  v7 = sub_3909460(*(_QWORD *)(a1 + 8));
  if ( *(_DWORD *)v7 == 2 )
  {
    v11 = *(_QWORD *)(v7 + 8);
    v9 = *(_QWORD *)(v7 + 16);
  }
  else
  {
    v8 = *(_QWORD *)(v7 + 16);
    v9 = 0;
    if ( v8 )
    {
      v10 = v8 - 1;
      if ( v8 == 1 )
        v10 = 1;
      if ( v10 <= v8 )
        v8 = v10;
      v9 = v8 - 1;
      v8 = 1;
    }
    v11 = *(_QWORD *)(v7 + 8) + v8;
  }
  v12 = -1073741760;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 25 )
    goto LABEL_14;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 3 )
  {
    v53 = 1;
    v4 = "expected string in directive";
    goto LABEL_3;
  }
  v16 = sub_3909460(*(_QWORD *)(a1 + 8));
  v17 = *(_QWORD *)(v16 + 16);
  v18 = v16;
  v19 = v17 - 1;
  if ( !v17 )
  {
    (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8), 0, v18);
    goto LABEL_85;
  }
  v20 = *(_QWORD *)(a1 + 8);
  v21 = *(_QWORD *)(v18 + 8);
  if ( v17 == 1 )
    v19 = 1;
  if ( v19 <= v17 )
    v17 = v19;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v20 + 136LL))(v20);
  v23 = (_BYTE *)(v21 + v17);
  if ( v23 == (_BYTE *)(v21 + 1) )
    goto LABEL_85;
  v24 = (_BYTE *)(v21 + 1);
  v25 = 0;
  v26 = 0;
  while ( 2 )
  {
    switch ( *v24 )
    {
      case 'D':
        BYTE1(v26) |= 1u;
        goto LABEL_36;
      case 'a':
        goto LABEL_36;
      case 'b':
        if ( (v26 & 8) != 0 )
          goto LABEL_92;
        v26 = v26 & 0xFFFFFFFA | 1;
        goto LABEL_36;
      case 'd':
        if ( (v26 & 1) == 0 )
        {
          v41 = v26;
          LOBYTE(v41) = v26 & 0x7F;
          v42 = v41;
          v43 = v41 | 0xC;
          v22 = v42 | 8;
          v32 = (v26 & 0x20) == 0;
          v26 = v22;
          if ( v32 )
            v26 = v43;
LABEL_36:
          if ( v23 == ++v24 )
          {
            if ( v26 )
            {
              v12 = v26 & 2;
              if ( (v26 & 2) != 0 )
                v12 = 536870944;
              v33 = v26 & 5;
              v34 = v26 & 0x20;
              v35 = v26 & 0x100;
              v36 = v26 & 0x40;
              v37 = v26 & 0x80;
              v38 = v26 & 0x10;
              if ( (v26 & 8) != 0 )
                v12 |= 0x40u;
              v39 = v12;
              if ( v33 == 1 )
              {
                LOBYTE(v39) = v12 | 0x80;
                v12 = v39;
              }
              if ( v34 )
                v12 |= 0x800u;
              if ( v35 )
              {
LABEL_47:
                v12 |= 0x2000000u;
                goto LABEL_48;
              }
LABEL_86:
              if ( v9 > 5 && *(_DWORD *)v11 == 1650811950 && *(_WORD *)(v11 + 4) == 26485 )
                goto LABEL_47;
LABEL_48:
              if ( !v36 )
                v12 |= 0x40000000u;
              if ( !v37 )
                v12 |= 0x80000000;
              if ( v38 )
                v12 |= 0x10000000u;
LABEL_14:
              v13 = *(_QWORD *)(a1 + 8);
              v48 = 0;
              v49 = 0;
              v50 = 0;
              if ( **(_DWORD **)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v13 + 40LL))(v13) + 8) != 25 )
              {
LABEL_15:
                if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8))
                                 + 8) == 9 )
                {
                  if ( (v12 & 0x20000000) != 0 )
                  {
                    v14 = 1;
                    v15 = *(_DWORD *)(*(_QWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8))
                                                + 32)
                                    + 728LL);
                    if ( v15 == 29 || v15 == 1 )
                      v12 |= 0x20000u;
                  }
                  else
                  {
                    v47 = 17;
                    if ( (v12 & 0x40000000) != 0 )
                      v47 = ((v12 >> 31) & 0xE) + 3;
                    v14 = v47;
                  }
                  sub_38FFA70(a1, (_BYTE *)v11, v9, v12, v14, v48, v49, v50);
                  return 0;
                }
                v53 = 1;
                v4 = "unexpected token in directive";
LABEL_3:
                v5 = *(_QWORD *)(a1 + 8);
                v51[0] = v4;
                v52 = 3;
                return sub_3909CF0(v5, v51, 0, 0, v2, v3);
              }
              v46 = *(_QWORD *)(a1 + 8);
              v48 = 2;
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v46 + 136LL))(v46);
              if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8))
                               + 8) != 2 )
              {
                v53 = 1;
                v4 = "expected comdat type such as 'discard' or 'largest' after protection bits";
                goto LABEL_3;
              }
              result = sub_3900970(a1, (__int64)&v48);
              if ( (_BYTE)result )
                return result;
              if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8))
                               + 8) != 25 )
              {
                v53 = 1;
                v4 = "expected comma in directive";
                goto LABEL_3;
              }
              v12 |= 0x1000u;
              (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
              if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 8) + 144LL))(
                      *(_QWORD *)(a1 + 8),
                      &v49) )
                goto LABEL_15;
LABEL_2:
              v53 = 1;
              v4 = "expected identifier in directive";
              goto LABEL_3;
            }
LABEL_85:
            v12 = 64;
            v36 = 0;
            v37 = 0;
            v38 = 0;
            goto LABEL_86;
          }
          continue;
        }
LABEL_92:
        v53 = 1;
        v27 = "conflicting section flags 'b' and 'd'.";
LABEL_32:
        v28 = *(_QWORD *)(a1 + 8);
        v51[0] = v27;
        v52 = 3;
        result = sub_3909CF0(v28, v51, 0, 0, v25, v22);
        if ( !(_BYTE)result )
        {
          v12 = -1073741760;
          goto LABEL_14;
        }
        return result;
      case 'n':
        v26 = v26 & 0xFFFFFFDB | 0x20;
        goto LABEL_36;
      case 'r':
        v45 = v26;
        v44 = v26;
        LOBYTE(v45) = v26 | 0x80;
        if ( (v26 & 2) == 0 )
        {
          LOBYTE(v44) = v26 | 0x88;
          v45 = v44;
        }
        v25 = 0;
        v26 = v45;
        if ( (v45 & 0x20) == 0 )
          v26 = v45 | 4;
        goto LABEL_36;
      case 's':
        v29 = v26;
        LOBYTE(v29) = v26 & 0x7F;
        v30 = v29;
        v31 = v29 | 0x1C;
        v22 = v30 | 0x18;
        v32 = (v26 & 0x20) == 0;
        v26 = v22;
        if ( v32 )
          v26 = v31;
        goto LABEL_36;
      case 'w':
        LOBYTE(v26) = v26 & 0x7F;
        v25 = 1;
        goto LABEL_36;
      case 'x':
        v40 = v26 | 2;
        v22 = v26 | 6;
        if ( (v26 & 0x20) == 0 )
          v40 = v26 | 6;
        v26 = v40;
        LOBYTE(v40) = v40 | 0x80;
        if ( !(_BYTE)v25 )
          v26 = v40;
        goto LABEL_36;
      case 'y':
        LOBYTE(v26) = v26 | 0xC0;
        goto LABEL_36;
      default:
        v53 = 1;
        v27 = "unknown flag";
        goto LABEL_32;
    }
  }
}
