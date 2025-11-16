// Function: sub_2792F80
// Address: 0x2792f80
//
__int64 __fastcall sub_2792F80(__int64 a1, __int64 a2)
{
  _BYTE *v2; // rdx
  unsigned int v4; // esi
  __int64 v5; // rdi
  unsigned int v6; // r8d
  unsigned int v7; // ecx
  __int64 v8; // rax
  _BYTE *v9; // r9
  unsigned int v10; // r13d
  int v12; // eax
  int v13; // ebx
  _QWORD *v14; // r9
  int v15; // r11d
  unsigned int v16; // ecx
  _QWORD *v17; // rax
  _BYTE *v18; // r10
  _DWORD *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  unsigned int v24; // eax
  unsigned int v25; // esi
  unsigned int v26; // r15d
  __int64 v27; // rcx
  __int64 v28; // r9
  __int64 *v29; // rdi
  int v30; // r11d
  unsigned int v31; // edx
  _QWORD *v32; // rax
  _BYTE *v33; // r8
  unsigned int *v34; // rax
  int v35; // r13d
  int v36; // r13d
  __int64 v37; // r13
  int v38; // eax
  int v39; // edx
  int v40; // eax
  int v41; // ecx
  int v42; // r10d
  _BYTE *v43; // [rsp+8h] [rbp-B8h] BYREF
  __int64 v44; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v45; // [rsp+18h] [rbp-A8h]
  unsigned __int64 v46[2]; // [rsp+20h] [rbp-A0h] BYREF
  _BYTE v47[16]; // [rsp+30h] [rbp-90h] BYREF
  __int64 v48; // [rsp+40h] [rbp-80h]
  _QWORD v49[2]; // [rsp+50h] [rbp-70h] BYREF
  char *v50; // [rsp+60h] [rbp-60h] BYREF
  char v51; // [rsp+70h] [rbp-50h] BYREF
  __int64 v52; // [rsp+80h] [rbp-40h]

  v2 = (_BYTE *)a2;
  v43 = (_BYTE *)a2;
  v4 = *(_DWORD *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  if ( v4 )
  {
    v6 = v4 - 1;
    v7 = (v4 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
    v8 = v5 + 16LL * v7;
    v9 = *(_BYTE **)v8;
    if ( v2 == *(_BYTE **)v8 )
    {
LABEL_3:
      if ( v8 != v5 + 16LL * v4 )
        return *(unsigned int *)(v8 + 8);
    }
    else
    {
      v12 = 1;
      while ( v9 != (_BYTE *)-4096LL )
      {
        v42 = v12 + 1;
        v7 = v6 & (v12 + v7);
        v8 = v5 + 16LL * v7;
        v9 = *(_BYTE **)v8;
        if ( v2 == *(_BYTE **)v8 )
          goto LABEL_3;
        v12 = v42;
      }
    }
    if ( *v2 <= 0x1Cu )
    {
      v13 = *(_DWORD *)(a1 + 208);
      v14 = 0;
      v15 = 1;
      v16 = v6 & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
      v17 = (_QWORD *)(v5 + 16LL * v16);
      v18 = (_BYTE *)*v17;
      if ( v2 == (_BYTE *)*v17 )
      {
LABEL_12:
        v19 = v17 + 1;
LABEL_13:
        *v19 = v13;
        v10 = *(_DWORD *)(a1 + 208);
        *(_DWORD *)(a1 + 208) = v10 + 1;
        return v10;
      }
      while ( v18 != (_BYTE *)-4096LL )
      {
        if ( v18 == (_BYTE *)-8192LL && !v14 )
          v14 = v17;
        v16 = v6 & (v15 + v16);
        v17 = (_QWORD *)(v5 + 16LL * v16);
        v18 = (_BYTE *)*v17;
        if ( v2 == (_BYTE *)*v17 )
          goto LABEL_12;
        ++v15;
      }
      if ( !v14 )
        v14 = v17;
      v40 = *(_DWORD *)(a1 + 16);
      ++*(_QWORD *)a1;
      v41 = v40 + 1;
      v49[0] = v14;
      if ( 4 * (v40 + 1) < 3 * v4 )
      {
        if ( v4 - *(_DWORD *)(a1 + 20) - v41 > v4 >> 3 )
        {
LABEL_51:
          *(_DWORD *)(a1 + 16) = v41;
          if ( *v14 != -4096 )
            --*(_DWORD *)(a1 + 20);
          *v14 = v2;
          v19 = v14 + 1;
          *((_DWORD *)v14 + 2) = 0;
          goto LABEL_13;
        }
LABEL_56:
        sub_D39D40(a1, v4);
        sub_22B1A50(a1, (__int64 *)&v43, v49);
        v2 = v43;
        v14 = (_QWORD *)v49[0];
        v41 = *(_DWORD *)(a1 + 16) + 1;
        goto LABEL_51;
      }
LABEL_55:
      v4 *= 2;
      goto LABEL_56;
    }
  }
  else if ( *v2 <= 0x1Cu )
  {
    ++*(_QWORD *)a1;
    v13 = *(_DWORD *)(a1 + 208);
    v49[0] = 0;
    goto LABEL_55;
  }
  v45 = 0;
  v44 = 4294967293LL;
  v46[0] = (unsigned __int64)v47;
  v46[1] = 0x400000000LL;
  v48 = 0;
  switch ( *v2 )
  {
    case ')':
    case '*':
    case '+':
    case ',':
    case '-':
    case '.':
    case '/':
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
    case ':':
    case ';':
    case 'C':
    case 'D':
    case 'E':
    case 'F':
    case 'G':
    case 'H':
    case 'I':
    case 'J':
    case 'K':
    case 'L':
    case 'M':
    case 'N':
    case 'O':
    case 'R':
    case 'S':
    case 'V':
    case 'Z':
    case '[':
    case '\\':
    case '^':
    case '`':
      sub_2793480(v49, a1);
      goto LABEL_15;
    case '?':
      sub_2794570(v49, a1);
      goto LABEL_15;
    case 'T':
      v36 = *(_DWORD *)(a1 + 208);
      *(_DWORD *)sub_10E84F0(a1, (__int64 *)&v43) = v36;
      v37 = (__int64)v43;
      *(_QWORD *)sub_2790A90(a1 + 120, (int *)(a1 + 208)) = v37;
      v10 = *(_DWORD *)(a1 + 208);
      *(_DWORD *)(a1 + 208) = v10 + 1;
      goto LABEL_21;
    case 'U':
      v10 = sub_27938B0(a1, v2);
      goto LABEL_21;
    case ']':
      sub_2794270(v49, a1);
LABEL_15:
      LODWORD(v44) = v49[0];
      BYTE4(v44) = BYTE4(v49[0]);
      v45 = v49[1];
      sub_2789850((__int64)v46, &v50, v20, v21, v22, v23);
      v48 = v52;
      if ( v50 != &v51 )
        _libc_free((unsigned __int64)v50);
      v24 = sub_2792D30(a1, (__int64)&v44);
      v25 = *(_DWORD *)(a1 + 24);
      v26 = v24;
      v10 = v24;
      if ( !v25 )
      {
        ++*(_QWORD *)a1;
        v49[0] = 0;
LABEL_58:
        v25 *= 2;
        goto LABEL_59;
      }
      v27 = (__int64)v43;
      v28 = *(_QWORD *)(a1 + 8);
      v29 = 0;
      v30 = 1;
      v31 = (v25 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
      v32 = (_QWORD *)(v28 + 16LL * v31);
      v33 = (_BYTE *)*v32;
      if ( (_BYTE *)*v32 == v43 )
      {
LABEL_19:
        v34 = (unsigned int *)(v32 + 1);
        goto LABEL_20;
      }
      while ( v33 != (_BYTE *)-4096LL )
      {
        if ( !v29 && v33 == (_BYTE *)-8192LL )
          v29 = v32;
        v31 = (v25 - 1) & (v30 + v31);
        v32 = (_QWORD *)(v28 + 16LL * v31);
        v33 = (_BYTE *)*v32;
        if ( v43 == (_BYTE *)*v32 )
          goto LABEL_19;
        ++v30;
      }
      if ( !v29 )
        v29 = v32;
      v38 = *(_DWORD *)(a1 + 16);
      ++*(_QWORD *)a1;
      v39 = v38 + 1;
      v49[0] = v29;
      if ( 4 * (v38 + 1) >= 3 * v25 )
        goto LABEL_58;
      if ( v25 - *(_DWORD *)(a1 + 20) - v39 > v25 >> 3 )
        goto LABEL_38;
LABEL_59:
      sub_D39D40(a1, v25);
      sub_22B1A50(a1, (__int64 *)&v43, v49);
      v27 = (__int64)v43;
      v29 = (__int64 *)v49[0];
      v39 = *(_DWORD *)(a1 + 16) + 1;
LABEL_38:
      *(_DWORD *)(a1 + 16) = v39;
      if ( *v29 != -4096 )
        --*(_DWORD *)(a1 + 20);
      *v29 = v27;
      v34 = (unsigned int *)(v29 + 1);
      *((_DWORD *)v29 + 2) = 0;
LABEL_20:
      *v34 = v26;
LABEL_21:
      if ( (_BYTE *)v46[0] != v47 )
        _libc_free(v46[0]);
      return v10;
    default:
      v35 = *(_DWORD *)(a1 + 208);
      *(_DWORD *)sub_10E84F0(a1, (__int64 *)&v43) = v35;
      v10 = *(_DWORD *)(a1 + 208);
      *(_DWORD *)(a1 + 208) = v10 + 1;
      goto LABEL_21;
  }
}
