// Function: sub_908850
// Address: 0x908850
//
__int64 __fastcall sub_908850(__int64 a1)
{
  const char *v1; // r15
  size_t v2; // rbx
  __int64 v3; // rax
  _QWORD *v4; // r12
  _BYTE *v5; // rdi
  __int64 v6; // rdx
  size_t v7; // rsi
  __int64 v8; // r8
  _QWORD *v9; // rdi
  char *v10; // rdx
  size_t v11; // rax
  _BYTE *v12; // rdi
  size_t v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // r8
  _QWORD *v16; // rdi
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // rbx
  __int64 v23; // rcx
  _DWORD *v24; // r8
  char v25; // al
  __int64 v26; // rax
  __int64 v27; // rbx
  __int64 v28; // rax
  __int64 result; // rax
  size_t v30; // rax
  char v31; // dl
  char *v32; // rax
  __int64 i; // rdx
  __int64 v34; // r8
  size_t v35; // rdx
  size_t v36; // rdx
  char *v37; // rax
  char *v38; // rax
  char *v39; // [rsp+0h] [rbp-370h]
  _DWORD *v40; // [rsp+0h] [rbp-370h]
  _DWORD *v41; // [rsp+0h] [rbp-370h]
  _BYTE *v42; // [rsp+8h] [rbp-368h]
  _QWORD *v43; // [rsp+8h] [rbp-368h]
  void (__fastcall *v44)(__int64, __int64, _QWORD); // [rsp+8h] [rbp-368h]
  __int64 v45; // [rsp+8h] [rbp-368h]
  __int64 v46; // [rsp+8h] [rbp-368h]
  __int64 v47; // [rsp+8h] [rbp-368h]
  _QWORD *v48; // [rsp+10h] [rbp-360h] BYREF
  __int64 v49; // [rsp+18h] [rbp-358h]
  _QWORD v50[2]; // [rsp+20h] [rbp-350h] BYREF
  const char *v51; // [rsp+30h] [rbp-340h] BYREF
  __int64 v52; // [rsp+38h] [rbp-338h]
  __int16 v53; // [rsp+50h] [rbp-320h]
  _QWORD *v54; // [rsp+60h] [rbp-310h] BYREF
  size_t n; // [rsp+68h] [rbp-308h]
  _QWORD src[2]; // [rsp+70h] [rbp-300h] BYREF
  __int64 v57; // [rsp+80h] [rbp-2F0h]
  __int64 v58; // [rsp+88h] [rbp-2E8h]
  __int64 v59; // [rsp+90h] [rbp-2E0h]

  v48 = v50;
  v49 = 0;
  LOBYTE(v50[0]) = 0;
  sub_B6F950(a1, unk_4D04618 != 0);
  unk_4F6D2F8 = 0;
  v1 = (const char *)qword_4D046D8;
  if ( qword_4D046D8 )
  {
    if ( *qword_4D046D8 )
    {
      v30 = (size_t)&v1[strlen((const char *)qword_4D046D8) - 1];
      if ( v1 != (const char *)v30 )
      {
        while ( 1 )
        {
          v31 = *(_BYTE *)(v30 - 1);
          if ( v31 == 92 || v31 == 47 )
            break;
          if ( v1 == (const char *)--v30 )
            goto LABEL_3;
        }
        v1 = (const char *)v30;
      }
    }
  }
  else
  {
    v1 = "moduleOutput";
  }
LABEL_3:
  v2 = strlen(v1);
  v3 = sub_22077B0(880);
  v4 = (_QWORD *)v3;
  if ( v3 )
    sub_BA8740(v3, v1, v2, a1);
  v42 = v4 + 31;
  if ( unk_4F06A68 == 8 )
  {
    if ( unk_4D0461C )
      sub_2241130(
        &v48,
        0,
        v49,
        "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:1"
        "28-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64",
        167);
    else
      sub_2241130(
        &v48,
        0,
        v49,
        "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16"
        "-v32:32:32-v64:64:64-v128:128:128-n16:32:64",
        155);
    v52 = 19;
    v53 = 261;
    v51 = "nvptx64-nvidia-cuda";
  }
  else
  {
    sub_2241130(
      &v48,
      0,
      v49,
      "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v"
      "32:32:32-v64:64:64-v128:128:128-n16:32:64",
      155);
    v52 = 17;
    v53 = 261;
    v51 = "nvptx-nvidia-cuda";
  }
  sub_CC9F70(&v54, &v51);
  v5 = (_BYTE *)v4[29];
  if ( v54 == src )
  {
    v35 = n;
    if ( n )
    {
      if ( n == 1 )
        *v5 = src[0];
      else
        memcpy(v5, src, n);
      v35 = n;
      v5 = (_BYTE *)v4[29];
    }
    v4[30] = v35;
    v5[v35] = 0;
    v5 = v54;
  }
  else
  {
    v6 = src[0];
    v7 = n;
    if ( v5 == v42 )
    {
      v4[29] = v54;
      v4[30] = v7;
      v4[31] = v6;
    }
    else
    {
      v8 = v4[31];
      v4[29] = v54;
      v4[30] = v7;
      v4[31] = v6;
      if ( v5 )
      {
        v54 = v5;
        src[0] = v8;
        goto LABEL_13;
      }
    }
    v54 = src;
    v5 = src;
  }
LABEL_13:
  n = 0;
  *v5 = 0;
  v9 = v54;
  v4[33] = v57;
  v4[34] = v58;
  v4[35] = v59;
  if ( v9 != src )
    j_j___libc_free_0(v9, src[0] + 1LL);
  if ( unk_4D04630 )
  {
    v10 = off_4C5D110;
    v11 = 0;
    if ( off_4C5D110 )
    {
      v39 = off_4C5D110;
      v11 = strlen(off_4C5D110);
      v10 = v39;
    }
    v52 = v11;
    v53 = 261;
    v51 = v10;
    sub_CC9F70(&v54, &v51);
    v12 = (_BYTE *)v4[29];
    if ( v54 == src )
    {
      v36 = n;
      if ( n )
      {
        if ( n == 1 )
          *v12 = src[0];
        else
          memcpy(v12, src, n);
        v36 = n;
        v12 = (_BYTE *)v4[29];
      }
      v4[30] = v36;
      v12[v36] = 0;
      v12 = v54;
      goto LABEL_22;
    }
    v13 = n;
    v14 = src[0];
    if ( v12 == v42 )
    {
      v4[29] = v54;
      v4[30] = v13;
      v4[31] = v14;
    }
    else
    {
      v15 = v4[31];
      v4[29] = v54;
      v4[30] = v13;
      v4[31] = v14;
      if ( v12 )
      {
        v54 = v12;
        src[0] = v15;
        goto LABEL_22;
      }
    }
    v54 = src;
    v12 = src;
LABEL_22:
    n = 0;
    *v12 = 0;
    v16 = v54;
    v4[33] = v57;
    v4[34] = v58;
    v4[35] = v59;
    if ( v16 != src )
      j_j___libc_free_0(v16, src[0] + 1LL);
    goto LABEL_24;
  }
  sub_BA9520(v4, v48, v49);
LABEL_24:
  v17 = v49;
  v43 = v48;
  v18 = sub_22077B0(496);
  v19 = v18;
  if ( v18 )
    sub_AE3F70(v18, v43, v17);
  sub_909E90(&v54, v4, v19);
  if ( unk_4D04610 )
    goto LABEL_54;
  v20 = qword_4F07288;
  v21 = *(_QWORD *)(qword_4F07288 + 104);
  if ( v21 )
  {
    do
    {
      while ( (unsigned int)sub_736DD0(v21) )
      {
        v21 = *(_QWORD *)(v21 + 112);
        if ( !v21 )
          goto LABEL_32;
      }
      sub_91CA00(v21);
      v21 = *(_QWORD *)(v21 + 112);
    }
    while ( v21 );
LABEL_32:
    v20 = qword_4F07288;
  }
  v22 = *(_QWORD *)(v20 + 112);
  if ( !v22 )
    goto LABEL_45;
  do
  {
    if ( (*(_BYTE *)(v22 + 170) & 0x60) == 0
      && *(_BYTE *)(v22 + 177) != 5
      && ((*(_BYTE *)(v22 + 176) & 0x40) == 0 || *(_BYTE *)(*(_QWORD *)(v22 + 120) + 140LL) != 14)
      && (unk_4D04630
       || sub_5E3770(v22)
       && ((unsigned __int8)sub_91C2E0(*(_QWORD *)(v22 + 120)) || (*(_BYTE *)(v22 + 156) & 1) != 0)
       && (*(_BYTE *)(v22 - 8) & 0x10) == 0) )
    {
      for ( i = *(_QWORD *)(v22 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      if ( (*(_BYTE *)(v22 + 156) & 2) == 0
        || *(_QWORD *)(i + 128)
        || (v47 = i, !sub_8D3410(i))
        || *(_QWORD *)(v47 + 176)
        || (*(_BYTE *)(v47 + 169) & 0x20) != 0 )
      {
        if ( !HIDWORD(qword_4D045BC) )
        {
          if ( *(_BYTE *)(v22 + 136) != 1 )
          {
            v34 = v22 + 64;
LABEL_85:
            *(_BYTE *)(v22 + 136) = 2;
LABEL_86:
            sub_91CAC0(v34);
            sub_916690(&v54, v22, 0);
            goto LABEL_43;
          }
          if ( !sub_8D23B0(*(_QWORD *)(v22 + 120)) )
          {
            v34 = v22 + 64;
            if ( *(_BYTE *)(v22 + 136) == 1 && (*(_BYTE *)(v22 - 8) & 0x10) == 0 && (*(_BYTE *)(v22 + 169) & 0x30) != 0 )
            {
              v37 = (char *)sub_91B6B0(v22);
              v38 = sub_693CD0(v37);
              sub_684B10(0xDADu, (_DWORD *)(v22 + 64), (__int64)v38);
              v34 = v22 + 64;
            }
            goto LABEL_85;
          }
        }
      }
      if ( *(_BYTE *)(v22 + 136) != 1 )
      {
        v34 = v22 + 64;
        goto LABEL_86;
      }
    }
LABEL_43:
    v22 = *(_QWORD *)(v22 + 112);
  }
  while ( v22 );
  v20 = qword_4F07288;
LABEL_45:
  v23 = *(_QWORD *)(v20 + 144);
  if ( v23 )
  {
    v24 = (_DWORD *)&qword_4D045BC + 1;
    do
    {
      if ( !*v24 )
      {
        v25 = *(_BYTE *)(v23 + 198);
        if ( (v25 & 0x20) != 0 && (*(_DWORD *)(v23 + 192) & 0x8000400) == 0 && (*(_BYTE *)(v23 + 195) & 1) != 0 )
        {
          if ( *(_DWORD *)(v23 + 160) )
          {
            if ( (v25 & 0x10) != 0 )
              goto LABEL_73;
            goto LABEL_53;
          }
          if ( !unk_4D0471C )
            goto LABEL_53;
          v40 = v24;
          v45 = v23;
          v32 = sub_8258E0(v23, 0);
          sub_684B10(0xE99u, (_DWORD *)(v45 + 64), (__int64)v32);
          v24 = v40;
          v23 = v45;
        }
      }
      if ( *(_DWORD *)(v23 + 160) && (*(_BYTE *)(v23 + 198) & 0x10) != 0 )
      {
LABEL_73:
        if ( (unk_4D04630 || (*(_BYTE *)(v23 + 203) & 4) != 0 && (*(_BYTE *)(v23 - 8) & 0x10) == 0)
          && (*(_DWORD *)(v23 + 192) & 0x8000400) == 0 )
        {
          v41 = v24;
          v46 = v23;
          sub_9172F0(&v54, v23);
          v24 = v41;
          v23 = v46;
        }
      }
LABEL_53:
      v23 = *(_QWORD *)(v23 + 112);
    }
    while ( v23 );
  }
LABEL_54:
  sub_915400(&v54);
  v26 = sub_22077B0(16);
  v27 = v26;
  if ( v26 )
    sub_B848C0(v26);
  v44 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v27 + 16LL);
  v28 = sub_BE0980(0);
  v44(v27, v28, 0);
  if ( (unsigned __int8)sub_B89FE0(v27, v4) )
    sub_91B980("there was an error in verifying the lgenfe output!", 0);
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v27 + 8LL))(v27);
  if ( v19 )
  {
    sub_AE4030(v19);
    j_j___libc_free_0(v19, 496);
  }
  unk_4F6D2F8 = v4;
  result = sub_90A100(&v54);
  if ( v48 != v50 )
    return j_j___libc_free_0(v48, v50[0] + 1LL);
  return result;
}
