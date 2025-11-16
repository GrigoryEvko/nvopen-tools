// Function: sub_1A07EC0
// Address: 0x1a07ec0
//
__int64 __fastcall sub_1A07EC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  unsigned int v7; // r8d
  __int64 *v8; // r14
  __int64 v10; // r11
  __int64 v11; // r13
  __int64 *v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rax
  char v15; // dl
  char v16; // al
  unsigned int v17; // edx
  __int64 v18; // rsi
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rax
  unsigned int v21; // r15d
  int v22; // eax
  bool v23; // al
  __int64 v24; // rdi
  __int64 *v25; // rax
  unsigned int v26; // esi
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // r9
  unsigned int v30; // r15d
  int v31; // eax
  bool v32; // al
  int v33; // eax
  int v34; // eax
  __int64 v35; // rax
  __int64 v36; // rax
  int v37; // eax
  int v38; // eax
  __int64 v39; // [rsp+8h] [rbp-98h]
  __int64 v40; // [rsp+8h] [rbp-98h]
  __int64 v41; // [rsp+10h] [rbp-90h]
  __int64 v42; // [rsp+10h] [rbp-90h]
  unsigned int v43; // [rsp+18h] [rbp-88h]
  __int64 v44; // [rsp+18h] [rbp-88h]
  unsigned int v45; // [rsp+18h] [rbp-88h]
  __int64 v46; // [rsp+18h] [rbp-88h]
  __int64 v47; // [rsp+18h] [rbp-88h]
  const void **v48; // [rsp+20h] [rbp-80h]
  unsigned int v49; // [rsp+20h] [rbp-80h]
  int v50; // [rsp+28h] [rbp-78h]
  __int64 v51; // [rsp+28h] [rbp-78h]
  unsigned __int8 v54; // [rsp+38h] [rbp-68h]
  unsigned __int8 v55; // [rsp+38h] [rbp-68h]
  __int64 v56; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v57; // [rsp+48h] [rbp-58h]
  __int64 v58; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v59; // [rsp+58h] [rbp-48h]
  __int64 v60; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v61; // [rsp+68h] [rbp-38h]

  v7 = 0;
  v8 = *(__int64 **)(a3 + 8);
  if ( v8 != *(__int64 **)(a4 + 8) )
    return v7;
  v10 = a2;
  v11 = a3;
  v12 = (__int64 *)a4;
  v13 = *(_QWORD *)(*(_QWORD *)a3 + 8LL);
  if ( v13 )
    v50 = (*(_QWORD *)(v13 + 8) == 0) + 1;
  else
    v50 = 1;
  v14 = *(_QWORD *)(*(_QWORD *)a4 + 8LL);
  if ( v14 )
    v50 += *(_QWORD *)(v14 + 8) == 0;
  v15 = *(_BYTE *)(a3 + 36);
  v16 = *(_BYTE *)(a4 + 36);
  if ( v15 != v16 )
  {
    if ( v16 )
    {
      v25 = (__int64 *)v11;
      v11 = a4;
      v12 = v25;
    }
    v17 = *(_DWORD *)(v11 + 24);
    v48 = (const void **)(v11 + 16);
    v59 = v17;
    if ( v17 > 0x40 )
    {
      sub_16A4FD0((__int64)&v58, v48);
      v17 = v59;
      v10 = a2;
      if ( v59 > 0x40 )
      {
        sub_16A8F40(&v58);
        v17 = v59;
        v19 = v58;
        v59 = 0;
        v10 = a2;
        v61 = v17;
        v60 = v58;
        if ( v17 > 0x40 )
        {
          sub_16A8F00(&v60, v12 + 2);
          v17 = v61;
          v10 = a2;
          v57 = v61;
          v56 = v60;
          if ( v59 > 0x40 && v58 )
          {
            j_j___libc_free_0_0(v58);
            v17 = v57;
            v10 = a2;
          }
          if ( v17 > 0x40 )
          {
            v41 = v10;
            v45 = v17;
            v33 = sub_16A57B0((__int64)&v56);
            v10 = v41;
            if ( v45 == v33 )
              goto LABEL_24;
            v34 = sub_16A58F0((__int64)&v56);
            v17 = v45;
            v10 = v41;
            LOBYTE(v7) = v45 == v34;
LABEL_16:
            if ( !(_BYTE)v7 )
            {
              v21 = *(_DWORD *)(a5 + 8);
              if ( v21 <= 0x40 )
              {
                v23 = *(_QWORD *)a5 == 0;
              }
              else
              {
                v39 = v10;
                v43 = v17;
                v22 = sub_16A57B0(a5);
                v10 = v39;
                v7 = 0;
                v17 = v43;
                v23 = v21 == v22;
              }
              if ( v50 <= 1 && v23 )
              {
                if ( v17 > 0x40 && v56 )
                {
                  v54 = v7;
                  j_j___libc_free_0_0(v56);
                  return v54;
                }
                return v7;
              }
            }
LABEL_24:
            *a6 = sub_19FF2B0(v10, v8, (__int64)&v56);
            if ( *(_DWORD *)(a5 + 8) > 0x40u )
              sub_16A8F00((__int64 *)a5, (__int64 *)v48);
            else
              *(_QWORD *)a5 ^= *(_QWORD *)(v11 + 16);
            if ( v57 <= 0x40 )
              goto LABEL_29;
            v24 = v56;
            if ( !v56 )
              goto LABEL_29;
LABEL_28:
            j_j___libc_free_0_0(v24);
LABEL_29:
            if ( *(_BYTE *)(*(_QWORD *)v11 + 16LL) > 0x17u )
            {
              v60 = *(_QWORD *)v11;
              sub_1A062A0(a1 + 64, &v60);
            }
            v7 = 1;
            if ( *(_BYTE *)(*v12 + 16) > 0x17u )
            {
              v60 = *v12;
              sub_1A062A0(a1 + 64, &v60);
              return 1;
            }
            return v7;
          }
LABEL_14:
          if ( !v56 )
            goto LABEL_24;
          LOBYTE(v7) = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v17) == v56;
          goto LABEL_16;
        }
LABEL_13:
        v20 = v12[2] ^ v19;
        v57 = v17;
        v56 = v20;
        goto LABEL_14;
      }
      v18 = v58;
    }
    else
    {
      v18 = *(_QWORD *)(v11 + 16);
    }
    v19 = ~v18 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v17);
    goto LABEL_13;
  }
  v26 = *(_DWORD *)(v11 + 24);
  v7 = v11 + 16;
  v61 = v26;
  if ( !v15 )
  {
    if ( v26 > 0x40 )
    {
      v51 = v10;
      sub_16A4FD0((__int64)&v60, (const void **)(v11 + 16));
      v26 = v61;
      v10 = v51;
      if ( v61 > 0x40 )
      {
        sub_16A8F00(&v60, v12 + 2);
        v26 = v61;
        v36 = v60;
        v10 = v51;
        goto LABEL_65;
      }
      v35 = v60;
    }
    else
    {
      v35 = *(_QWORD *)(v11 + 16);
    }
    v36 = v12[2] ^ v35;
LABEL_65:
    v59 = v26;
    v58 = v36;
    *a6 = sub_19FF2B0(v10, v8, (__int64)&v58);
    goto LABEL_50;
  }
  if ( v26 <= 0x40 )
  {
    v27 = *(_QWORD *)(v11 + 16);
LABEL_37:
    v28 = v12[2] ^ v27;
    v59 = v26;
    v58 = v28;
    v29 = v28;
    goto LABEL_38;
  }
  v46 = v10;
  sub_16A4FD0((__int64)&v60, (const void **)(v11 + 16));
  v26 = v61;
  v10 = v46;
  if ( v61 <= 0x40 )
  {
    v27 = v60;
    goto LABEL_37;
  }
  sub_16A8F00(&v60, v12 + 2);
  v26 = v61;
  v29 = v60;
  v10 = v46;
  v59 = v61;
  v58 = v60;
  if ( v61 > 0x40 )
  {
    v42 = v46;
    v47 = v60;
    v49 = v61;
    v37 = sub_16A57B0((__int64)&v58);
    v10 = v42;
    if ( v49 != v37 )
    {
      v38 = sub_16A58F0((__int64)&v58);
      v26 = v49;
      v29 = v47;
      v10 = v42;
      LOBYTE(v7) = v49 == v38;
      goto LABEL_40;
    }
LABEL_48:
    *a6 = sub_19FF2B0(v10, v8, (__int64)&v58);
    if ( *(_DWORD *)(a5 + 8) > 0x40u )
      sub_16A8F00((__int64 *)a5, &v58);
    else
      *(_QWORD *)a5 ^= v58;
LABEL_50:
    if ( v59 <= 0x40 )
      goto LABEL_29;
    v24 = v58;
    if ( !v58 )
      goto LABEL_29;
    goto LABEL_28;
  }
LABEL_38:
  if ( !v29 )
    goto LABEL_48;
  LOBYTE(v7) = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v26) == v29;
LABEL_40:
  if ( (_BYTE)v7 )
    goto LABEL_48;
  v30 = *(_DWORD *)(a5 + 8);
  if ( v30 <= 0x40 )
  {
    v32 = *(_QWORD *)a5 == 0;
  }
  else
  {
    v40 = v10;
    v44 = v29;
    v31 = sub_16A57B0(a5);
    v10 = v40;
    v7 = 0;
    v29 = v44;
    v32 = v30 == v31;
  }
  if ( v50 > 1 || !v32 )
    goto LABEL_48;
  if ( v26 > 0x40 && v29 )
  {
    v55 = v7;
    j_j___libc_free_0_0(v29);
    return v55;
  }
  return v7;
}
