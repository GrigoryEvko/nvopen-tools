// Function: sub_EC44F0
// Address: 0xec44f0
//
__int64 __fastcall sub_EC44F0(__int64 a1, __int64 a2)
{
  const char *v3; // rax
  __int64 v4; // rdi
  __int64 result; // rax
  __int64 v6; // rax
  size_t v7; // r13
  __int64 v8; // r15
  __int64 v9; // r13
  unsigned int v10; // r14d
  __int64 v11; // rdi
  int v12; // eax
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // rdx
  __int64 v16; // rbx
  int v17; // r8d
  int v18; // edx
  int v19; // ecx
  int v20; // edi
  unsigned int v21; // eax
  _BYTE *v22; // rbx
  _BYTE *v23; // r14
  _BYTE *v24; // rdx
  char v25; // di
  unsigned int v26; // esi
  const char *v27; // rax
  __int64 v28; // rdi
  int v29; // eax
  int v30; // r10d
  int v31; // r9d
  int v32; // eax
  unsigned int v33; // esi
  unsigned int v34; // eax
  unsigned int v35; // r8d
  int v36; // eax
  bool v37; // zf
  unsigned int v38; // eax
  unsigned int v39; // r8d
  int v40; // eax
  unsigned int v41; // edi
  unsigned int v42; // eax
  __int64 v43; // rdi
  unsigned __int8 v44; // [rsp+Fh] [rbp-71h] BYREF
  __int64 v45; // [rsp+10h] [rbp-70h] BYREF
  __int64 v46; // [rsp+18h] [rbp-68h]
  _QWORD v47[4]; // [rsp+20h] [rbp-60h] BYREF
  char v48; // [rsp+40h] [rbp-40h]
  char v49; // [rsp+41h] [rbp-3Fh]

  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 2
    && **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 3 )
  {
    goto LABEL_3;
  }
  v6 = sub_ECD7B0(*(_QWORD *)(a1 + 8));
  if ( *(_DWORD *)v6 == 2 )
  {
    v8 = *(_QWORD *)(v6 + 8);
    v7 = *(_QWORD *)(v6 + 16);
  }
  else
  {
    v7 = *(_QWORD *)(v6 + 16);
    v8 = *(_QWORD *)(v6 + 8);
    if ( v7 )
    {
      v9 = v7 - 1;
      if ( !v9 )
        v9 = 1;
      ++v8;
      v7 = v9 - 1;
    }
  }
  v10 = -1073741760;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 26 )
    goto LABEL_12;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 3 )
  {
    v49 = 1;
    v3 = "expected string in directive";
    goto LABEL_4;
  }
  v13 = sub_ECD7B0(*(_QWORD *)(a1 + 8));
  v14 = *(_QWORD *)(v13 + 16);
  v15 = *(_QWORD *)(v13 + 8);
  if ( !v14 || (v16 = v14 - 1) == 0 )
  {
    (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8), a2, v15);
    goto LABEL_25;
  }
  v22 = (_BYTE *)(v15 + v16);
  v23 = (_BYTE *)(v15 + 1);
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  if ( v22 == v23 )
    goto LABEL_25;
  v24 = v23;
  v25 = 0;
  v26 = 0;
  while ( 2 )
  {
    switch ( *v24 )
    {
      case 'D':
        v26 |= 0x100u;
        goto LABEL_47;
      case 'a':
        goto LABEL_47;
      case 'b':
        if ( (v26 & 8) != 0 )
          goto LABEL_85;
        v26 = v26 & 0xFFFFFFFA | 1;
        goto LABEL_47;
      case 'd':
        if ( (v26 & 1) == 0 )
        {
          v34 = v26;
          LOBYTE(v34) = v26 & 0x7F;
          v35 = v34;
          v36 = v34 | 0xC;
          v37 = (v26 & 0x20) == 0;
          v26 = v35 | 8;
          if ( v37 )
            v26 = v36;
LABEL_47:
          if ( v22 == ++v24 )
          {
            if ( v26 )
            {
              v10 = v26 & 2;
              if ( (v26 & 2) != 0 )
                v10 = 536870944;
              v30 = v26 & 5;
              v31 = v26 & 0x20;
              v32 = v26 & 0x100;
              v17 = v26 & 0x40;
              v19 = v26 & 0x10;
              v20 = v26 & 0x80;
              v18 = v26 & 0x200;
              if ( (v26 & 8) != 0 )
                v10 |= 0x40u;
              v33 = v10;
              if ( v30 == 1 )
              {
                LOBYTE(v33) = v10 | 0x80;
                v10 = v33;
              }
              if ( v31 )
                v10 |= 0x800u;
              if ( v32 )
              {
LABEL_58:
                v10 |= 0x2000000u;
                goto LABEL_28;
              }
LABEL_26:
              if ( v7 > 5 && *(_DWORD *)v8 == 1650811950 && *(_WORD *)(v8 + 4) == 26485 )
                goto LABEL_58;
LABEL_28:
              if ( !v17 )
                v10 |= 0x40000000u;
              if ( !v20 )
                v10 |= 0x80000000;
              if ( v19 )
                v10 |= 0x10000000u;
              v21 = v10;
              if ( v18 )
              {
                BYTE1(v21) = BYTE1(v10) | 2;
                v10 = v21;
              }
LABEL_12:
              v11 = *(_QWORD *)(a1 + 8);
              v44 = 0;
              v45 = 0;
              v46 = 0;
              if ( **(_DWORD **)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL))(v11) + 8) != 26 )
              {
LABEL_13:
                if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8))
                                 + 8) == 9 )
                {
                  if ( (v10 & 0x20) != 0 )
                  {
                    v12 = *(_DWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8))
                                    + 56);
                    if ( v12 == 36 || v12 == 1 )
                      v10 |= 0x20000u;
                  }
                  sub_EC2EA0(a1, (_BYTE *)v8, v7, v10, v45, v46, v44);
                  return 0;
                }
                v49 = 1;
                v3 = "unexpected token in directive";
LABEL_4:
                v4 = *(_QWORD *)(a1 + 8);
                v47[0] = v3;
                v48 = 3;
                return sub_ECE0E0(v4, v47, 0, 0);
              }
              v43 = *(_QWORD *)(a1 + 8);
              v44 = 2;
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v43 + 184LL))(v43);
              if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8))
                               + 8) != 2 )
              {
                v49 = 1;
                v3 = "expected comdat type such as 'discard' or 'largest' after protection bits";
                goto LABEL_4;
              }
              result = sub_EC3DC0(a1, (char *)&v44);
              if ( (_BYTE)result )
                return result;
              if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8))
                               + 8) != 26 )
              {
                v49 = 1;
                v3 = "expected comma in directive";
                goto LABEL_4;
              }
              v10 |= 0x1000u;
              (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
              if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 8) + 192LL))(
                      *(_QWORD *)(a1 + 8),
                      &v45) )
                goto LABEL_13;
LABEL_3:
              v49 = 1;
              v3 = "expected identifier in directive";
              goto LABEL_4;
            }
LABEL_25:
            v10 = 64;
            v17 = 0;
            v18 = 0;
            v19 = 0;
            v20 = 0;
            goto LABEL_26;
          }
          continue;
        }
LABEL_85:
        v49 = 1;
        v27 = "conflicting section flags 'b' and 'd'.";
LABEL_41:
        v28 = *(_QWORD *)(a1 + 8);
        v47[0] = v27;
        v48 = 3;
        result = sub_ECE0E0(v28, v47, 0, 0);
        if ( !(_BYTE)result )
        {
          v10 = -1073741760;
          goto LABEL_12;
        }
        return result;
      case 'i':
        v26 |= 0x200u;
        goto LABEL_47;
      case 'n':
        v26 = v26 & 0xFFFFFFDB | 0x20;
        goto LABEL_47;
      case 'r':
        v42 = v26;
        v41 = v26;
        LOBYTE(v42) = v26 | 0x80;
        if ( (v26 & 2) == 0 )
        {
          LOBYTE(v41) = v26 | 0x88;
          v42 = v41;
        }
        v25 = 0;
        v26 = v42;
        if ( (v42 & 0x20) == 0 )
          v26 = v42 | 4;
        goto LABEL_47;
      case 's':
        v38 = v26;
        LOBYTE(v38) = v26 & 0x7F;
        v39 = v38;
        v40 = v38 | 0x1C;
        v37 = (v26 & 0x20) == 0;
        v26 = v39 | 0x18;
        if ( v37 )
          v26 = v40;
        goto LABEL_47;
      case 'w':
        LOBYTE(v26) = v26 & 0x7F;
        v25 = 1;
        goto LABEL_47;
      case 'x':
        v29 = v26 | 2;
        if ( (v26 & 0x20) == 0 )
          v29 = v26 | 6;
        v26 = v29;
        LOBYTE(v29) = v29 | 0x80;
        if ( !v25 )
          v26 = v29;
        goto LABEL_47;
      case 'y':
        LOBYTE(v26) = v26 | 0xC0;
        goto LABEL_47;
      default:
        v49 = 1;
        v27 = "unknown flag";
        goto LABEL_41;
    }
  }
}
