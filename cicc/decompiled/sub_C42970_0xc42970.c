// Function: sub_C42970
// Address: 0xc42970
//
__int64 __fastcall sub_C42970(__int64 a1, __int64 a2, unsigned __int8 *a3, __int64 a4, char a5)
{
  unsigned __int8 *v8; // rbx
  int v9; // eax
  unsigned __int64 v10; // rsi
  int v11; // r15d
  __int64 v12; // rcx
  __int64 v13; // rdi
  unsigned __int64 v14; // rdx
  char v15; // al
  unsigned __int8 *v16; // r11
  unsigned __int8 *v17; // rax
  unsigned __int8 *v18; // r15
  unsigned __int8 *v19; // r8
  int v20; // r10d
  unsigned int v21; // eax
  __int64 v22; // rdx
  __int64 v23; // r13
  unsigned int v24; // r12d
  __int64 v25; // rax
  __int64 v26; // rbx
  const char *v27; // rax
  unsigned int v28; // eax
  __int64 v29; // rdx
  __int64 v30; // r12
  unsigned int v31; // r13d
  __int64 v32; // rax
  unsigned __int8 *v34; // rax
  unsigned __int8 *v35; // rdx
  __int64 v36; // rsi
  int v37; // eax
  unsigned int v38; // eax
  const char *v39; // rax
  unsigned int v40; // eax
  __int64 v41; // rdx
  __int64 v42; // r12
  unsigned int v43; // r13d
  __int64 v44; // rax
  __int64 v45; // rbx
  int v46; // eax
  int v47; // eax
  int v48; // r8d
  char v49; // al
  int v50; // [rsp+4h] [rbp-9Ch]
  int v51; // [rsp+8h] [rbp-98h]
  __int64 v53; // [rsp+18h] [rbp-88h]
  unsigned __int8 *v54; // [rsp+28h] [rbp-78h] BYREF
  unsigned __int8 *v55; // [rsp+30h] [rbp-70h] BYREF
  char v56; // [rsp+38h] [rbp-68h]
  _QWORD v57[4]; // [rsp+40h] [rbp-60h] BYREF
  char v58; // [rsp+60h] [rbp-40h]
  char v59; // [rsp+61h] [rbp-3Fh]

  v8 = &a3[a4];
  *(_BYTE *)(a2 + 20) = *(_BYTE *)(a2 + 20) & 0xF8 | 2;
  sub_C33EE0(a2);
  *(_DWORD *)(a2 + 16) = 0;
  v53 = sub_C33900(a2);
  v9 = sub_C337D0(a2);
  v10 = (unsigned __int64)a3;
  v11 = v9;
  sub_C32010((__int64)&v55, a3, v8, &v54);
  v13 = v56 & 1;
  v14 = (unsigned int)(2 * v13);
  v15 = (2 * v13) | v56 & 0xFD;
  v56 = v15;
  if ( (_BYTE)v13 )
  {
    *(_BYTE *)(a1 + 8) |= 3u;
    v56 = v15 & 0xFD;
    v34 = v55;
    v55 = 0;
    *(_QWORD *)a1 = (unsigned __int64)v34 & 0xFFFFFFFFFFFFFFFELL;
    goto LABEL_31;
  }
  v16 = v55;
  if ( v8 == v55 )
  {
LABEL_9:
    v59 = 1;
    v57[0] = "Hex strings require an exponent";
    v58 = 3;
    v21 = sub_C63BB0(v13, v10, v14, v12);
    v23 = v22;
    v24 = v21;
    v25 = sub_22077B0(64);
    v26 = v25;
    if ( v25 )
      sub_C63EB0(v25, v57, v24, v23);
    goto LABEL_15;
  }
  v17 = v55 + 1;
  v51 = 0;
  v50 = v11 << 6;
  v12 = (unsigned int)(v11 << 6);
  while ( 1 )
  {
    v14 = *(v17 - 1);
    v18 = v17 - 1;
    v19 = v17;
    if ( (_BYTE)v14 == 46 )
      break;
    v10 = (unsigned int)(__int16)word_3F64060[(unsigned __int8)v14];
    v20 = (__int16)word_3F64060[(unsigned __int8)v14];
    if ( v10 == 0xFFFFFFFF )
    {
      v14 = (unsigned int)v14 & 0xFFFFFFDF;
      if ( (_BYTE)v14 != 80 )
      {
        v59 = 1;
        v27 = "Invalid character in significand";
        goto LABEL_13;
      }
      if ( a3 == v18 )
        goto LABEL_56;
      v10 = (unsigned __int64)v54;
      if ( v54 != v8 )
      {
        if ( v18 - a3 != 1 )
        {
          if ( v16 != v18 )
            goto LABEL_42;
LABEL_63:
          v48 = sub_C36450(a2, a5, v51);
          v49 = *(_BYTE *)(a1 + 8);
          *(_DWORD *)a1 = v48;
          *(_BYTE *)(a1 + 8) = v49 & 0xFC | 2;
          goto LABEL_16;
        }
LABEL_56:
        v59 = 1;
        v27 = "Significand has no digits";
        goto LABEL_13;
      }
      if ( v16 == v18 )
        goto LABEL_63;
      v54 = v17 - 1;
      LODWORD(v10) = (_DWORD)v17 - 1;
LABEL_42:
      v35 = v17;
      v36 = (_DWORD)v10 - (_DWORD)v16 - ((unsigned int)((unsigned int)(v10 - (_DWORD)v16) < 0x80000000) - 1);
      if ( v17 != v8 )
      {
        v13 = *v17;
        if ( (((_BYTE)v13 - 43) & 0xFD) != 0 )
        {
          v37 = (char)v13;
LABEL_46:
          v12 = 0;
          while ( 1 )
          {
            v38 = v37 - 48;
            if ( v38 > 9 )
            {
              v59 = 1;
              v39 = "Invalid character in exponent";
              goto LABEL_52;
            }
            v12 = v38 + 10 * (_DWORD)v12;
            if ( (int)v12 > 0x7FFF )
              break;
            if ( v8 == ++v35 )
            {
              v46 = *(_DWORD *)(*(_QWORD *)a2 + 8LL) - v50 + 4 * v36 - 1;
              if ( (unsigned int)(v46 + 0x8000) <= 0xFFFF )
              {
                if ( (_BYTE)v13 == 45 )
                {
                  v47 = v46 - v12;
                  if ( (unsigned int)(v47 + 0x8000) >= 0x10000 )
                    v47 = -32768;
                }
                else
                {
                  v47 = v12 + v46;
                  if ( (unsigned int)(v47 + 0x8000) >= 0x10000 )
                    v47 = 0x7FFF;
                }
                goto LABEL_62;
              }
              break;
            }
            v37 = (char)*v35;
          }
          v47 = 0x7FFF;
          if ( (_BYTE)v13 == 45 )
            v47 = -32768;
LABEL_62:
          *(_DWORD *)(a2 + 16) = v47;
          goto LABEL_63;
        }
        v35 = v18 + 2;
        if ( v8 != v18 + 2 )
        {
          v37 = (char)v18[2];
          goto LABEL_46;
        }
      }
      v59 = 1;
      v39 = "Exponent has no digits";
LABEL_52:
      v57[0] = v39;
      v58 = 3;
      v40 = sub_C63BB0(v13, v36, v35, v12);
      v42 = v41;
      v43 = v40;
      v44 = sub_22077B0(64);
      v45 = v44;
      if ( v44 )
        sub_C63EB0(v44, v57, v43, v42);
      *(_BYTE *)(a1 + 8) |= 3u;
      *(_QWORD *)a1 = v45 & 0xFFFFFFFFFFFFFFFELL;
      goto LABEL_16;
    }
    if ( (_DWORD)v12 )
    {
      v12 = (unsigned int)(v12 - 4);
      v10 <<= v12;
      v14 = (unsigned int)v12 >> 6;
      *(_QWORD *)(v53 + 8 * v14) |= v10;
    }
    else if ( !(_BYTE)v13 )
    {
      if ( (unsigned int)v10 > 8 )
      {
        v51 = 3;
      }
      else
      {
        v14 = (unsigned int)(v10 - 1);
        if ( (unsigned int)v14 <= 6 )
        {
          v51 = 1;
        }
        else
        {
          if ( v8 == v17 )
            goto LABEL_68;
          v14 = (unsigned __int64)v17;
          while ( 1 )
          {
            v10 = *(unsigned __int8 *)v14;
            v13 = (unsigned __int8)(v10 - 46) & 0xFD;
            if ( (((_BYTE)v10 - 46) & 0xFD) != 0 )
              break;
            if ( v8 == (unsigned __int8 *)++v14 )
              goto LABEL_68;
          }
          if ( v8 == (unsigned __int8 *)v14 )
          {
LABEL_68:
            v59 = 1;
            v27 = "Invalid trailing hexadecimal fraction!";
            goto LABEL_13;
          }
          if ( word_3F64060[v10] == 0xFFFF )
            v10 = 2 * (unsigned int)(v20 != 0);
          else
            v10 = v20 == 0 ? 1 : 3;
          v51 = v10;
        }
      }
      v13 = 1;
    }
LABEL_8:
    ++v17;
    if ( v8 == v19 )
      goto LABEL_9;
  }
  if ( v54 == v8 )
  {
    v54 = v17 - 1;
    goto LABEL_8;
  }
  v59 = 1;
  v27 = "String contains multiple dots";
LABEL_13:
  v57[0] = v27;
  v58 = 3;
  v28 = sub_C63BB0(v13, v10, v14, v12);
  v30 = v29;
  v31 = v28;
  v32 = sub_22077B0(64);
  v26 = v32;
  if ( v32 )
    sub_C63EB0(v32, v57, v31, v30);
LABEL_15:
  *(_BYTE *)(a1 + 8) |= 3u;
  *(_QWORD *)a1 = v26 & 0xFFFFFFFFFFFFFFFELL;
LABEL_16:
  if ( (v56 & 2) != 0 )
    sub_C420F0(&v55);
  if ( (v56 & 1) != 0 )
  {
LABEL_31:
    if ( v55 )
      (*(void (__fastcall **)(unsigned __int8 *))(*(_QWORD *)v55 + 8LL))(v55);
  }
  return a1;
}
