// Function: sub_C7E470
// Address: 0xc7e470
//
__int64 __fastcall sub_C7E470(
        __int64 a1,
        unsigned int a2,
        const char **a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        unsigned int a7,
        unsigned int a8,
        unsigned __int16 a9)
{
  unsigned __int64 v13; // r12
  __int64 v14; // rbx
  __int64 v15; // rcx
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // r10
  int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  _QWORD *v26; // rax
  char *v27; // r15
  size_t v28; // r9
  char v29; // dl
  char v30; // al
  __int64 v31; // rax
  int v32; // eax
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  int v37; // eax
  int v38; // eax
  int v39; // eax
  __int64 v40; // rax
  int v41; // eax
  __int64 v42; // [rsp+0h] [rbp-D0h]
  __int64 v43; // [rsp+8h] [rbp-C8h]
  __int64 v44; // [rsp+10h] [rbp-C0h]
  __int64 v45; // [rsp+10h] [rbp-C0h]
  unsigned int v46; // [rsp+10h] [rbp-C0h]
  size_t n; // [rsp+18h] [rbp-B8h]
  _QWORD *v48; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v49[2]; // [rsp+30h] [rbp-A0h] BYREF
  char v50; // [rsp+40h] [rbp-90h]
  _OWORD v51[2]; // [rsp+50h] [rbp-80h] BYREF
  __int128 v52; // [rsp+70h] [rbp-60h]
  __int128 v53; // [rsp+80h] [rbp-50h]
  __int64 v54; // [rsp+90h] [rbp-40h]

  v13 = a5;
  v14 = a6;
  v15 = a8;
  if ( !byte_4F840F8 )
  {
    v37 = sub_2207590(&byte_4F840F8);
    v15 = a8;
    if ( v37 )
    {
      dword_4F84100 = sub_C7E1B0((__int64)&byte_4F840F8, a4);
      sub_2207640(&byte_4F840F8);
      v15 = a8;
    }
  }
  if ( v13 == -1 )
  {
    v13 = (unsigned __int64)a4;
    if ( a4 == (__int64 *)-1LL )
    {
      v46 = v15;
      v52 = 0;
      v54 = 0;
      HIDWORD(v52) = 0xFFFF;
      memset(v51, 0, sizeof(v51));
      v53 = 0;
      v39 = sub_C82AC0(a2);
      v15 = v46;
      if ( v39 )
      {
        *(_BYTE *)(a1 + 16) |= 1u;
        *(_DWORD *)a1 = v39;
        *(_QWORD *)(a1 + 8) = a3;
        return a1;
      }
      if ( DWORD2(v52) != 2 && DWORD2(v52) != 5 )
      {
        sub_C7DE70((__int64)v49, (__int64 *)a2, (__int64)a3);
        if ( (v50 & 1) != 0 )
        {
          v41 = v49[0];
          *(_BYTE *)(a1 + 16) |= 1u;
          *(_DWORD *)a1 = v41;
          *(_QWORD *)(a1 + 8) = v49[1];
        }
        else
        {
          v40 = v49[0];
          *(_BYTE *)(a1 + 16) &= ~1u;
          *(_QWORD *)a1 = v40;
        }
        return a1;
      }
      a4 = (__int64 *)v52;
      v13 = v52;
    }
  }
  v16 = a7;
  if ( ((unsigned __int8)v15 & (unsigned __int8)a7) == 0 && v13 > 0x3FFF )
  {
    a5 = (unsigned int)dword_4F84100;
    if ( v13 >= (unsigned int)dword_4F84100 )
    {
      if ( !(_BYTE)a7 || (v16 = a2, sub_C7D710(a2, (__int64)a4, v13, v14, dword_4F84100)) )
      {
        LODWORD(v51[0]) = 0;
        *((_QWORD *)&v51[0] + 1) = sub_2241E40(v16, a4, a3, v15, a5);
        v21 = sub_C7D7A0(48, a3, v17, v18, v19, v20);
        if ( !v21 )
          goto LABEL_29;
        v42 = v21;
        *(_QWORD *)v21 = off_4979C30;
        v43 = v21 + 24;
        v44 = v14 & (int)-sub_C85CF0();
        v22 = sub_C85CF0();
        sub_C82BB0(v43, a2, 0, v13 + (v14 & (v22 - 1)), v44, v51);
        v21 = v42;
        if ( !LODWORD(v51[0]) )
        {
          v45 = sub_C82270(v43);
          v38 = sub_C85CF0();
          sub_C7DA80(v42, v45 + (v14 & (v38 - 1)), v45 + (v14 & (v38 - 1)) + v13);
          v21 = v42;
          if ( !LODWORD(v51[0]) )
          {
LABEL_29:
            *(_BYTE *)(a1 + 16) &= ~1u;
            *(_QWORD *)a1 = v21;
            return a1;
          }
        }
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v21 + 8LL))(v21);
      }
    }
  }
  sub_C7DB40(&v48, v13, (__int64)a3, a9, a5, a6);
  v26 = v48;
  if ( !v48 )
  {
    v35 = sub_2241E50(&v48, v13, v23, v24, v25);
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = 12;
    *(_QWORD *)(a1 + 8) = v35;
    return a1;
  }
  v27 = (char *)v48[1];
  v28 = v48[2] - (_QWORD)v27;
  if ( !v28 )
    goto LABEL_35;
  while ( 1 )
  {
    n = v28;
    sub_C83650(v51, a2, v27, v28, v14);
    v29 = BYTE8(v51[0]) & 1;
    v30 = (2 * (BYTE8(v51[0]) & 1)) | BYTE8(v51[0]) & 0xFD;
    BYTE8(v51[0]) = v30;
    if ( v29 )
    {
      BYTE8(v51[0]) = v30 & 0xFD;
      v31 = *(_QWORD *)&v51[0];
      *(_QWORD *)&v51[0] = 0;
      v49[0] = v31 | 1;
      v32 = sub_C64300(v49, (__int64 *)a2);
      *(_BYTE *)(a1 + 16) |= 1u;
      *(_DWORD *)a1 = v32;
      v33 = v49[0];
      *(_QWORD *)(a1 + 8) = v34;
      if ( (v33 & 1) != 0 || (v33 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(v49, a2);
      if ( (BYTE8(v51[0]) & 2) == 0 )
      {
        if ( (BYTE8(v51[0]) & 1) != 0 && *(_QWORD *)&v51[0] )
          (*(void (__fastcall **)(_QWORD))(**(_QWORD **)&v51[0] + 8LL))(*(_QWORD *)&v51[0]);
        if ( v48 )
          (*(void (__fastcall **)(_QWORD *))(*v48 + 8LL))(v48);
        return a1;
      }
LABEL_45:
      sub_9CDF70(v51);
    }
    if ( !*(_QWORD *)&v51[0] )
      break;
    v27 += *(_QWORD *)&v51[0];
    v14 += *(_QWORD *)&v51[0];
    v28 = n - *(_QWORD *)&v51[0];
    if ( n == *(_QWORD *)&v51[0] )
      goto LABEL_34;
  }
  memset(v27, 0, n);
  if ( (BYTE8(v51[0]) & 2) != 0 )
    goto LABEL_45;
  if ( (BYTE8(v51[0]) & 1) != 0 && *(_QWORD *)&v51[0] )
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)&v51[0] + 8LL))(*(_QWORD *)&v51[0]);
LABEL_34:
  v26 = v48;
LABEL_35:
  *(_BYTE *)(a1 + 16) &= ~1u;
  *(_QWORD *)a1 = v26;
  return a1;
}
