// Function: sub_3071B90
// Address: 0x3071b90
//
__int64 __fastcall sub_3071B90(unsigned __int8 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  int v7; // r12d
  _QWORD *v8; // r8
  __int64 v9; // rax
  unsigned __int8 *v10; // rax
  __int64 v11; // rcx
  unsigned int v12; // r13d
  __int64 *v13; // rax
  __int64 v14; // r14
  __int64 v15; // r13
  unsigned __int8 **v16; // rax
  unsigned int v17; // esi
  __int64 *v19; // rax
  unsigned __int8 **v20; // rax
  unsigned __int64 v21; // r8
  char v22; // dl
  char v23; // r8
  __int64 v24; // rax
  int v25; // eax
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  _QWORD *v28; // [rsp+0h] [rbp-90h] BYREF
  __int64 v29; // [rsp+8h] [rbp-88h]
  _QWORD v30[4]; // [rsp+10h] [rbp-80h] BYREF
  __int64 v31; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int8 **v32; // [rsp+38h] [rbp-58h]
  __int64 v33; // [rsp+40h] [rbp-50h]
  int v34; // [rsp+48h] [rbp-48h]
  char v35; // [rsp+4Ch] [rbp-44h]
  unsigned __int8 *v36; // [rsp+50h] [rbp-40h] BYREF

  LODWORD(v6) = 1;
  v7 = 0;
  v8 = v30;
  v32 = &v36;
  v33 = 0x100000004LL;
  v28 = v30;
  v34 = 0;
  v35 = 1;
  v36 = a1;
  v31 = 1;
  v30[0] = a1;
  v29 = 0x400000001LL;
  while ( 1 )
  {
    v9 = (unsigned int)v6;
    v6 = (unsigned int)(v6 - 1);
    v10 = (unsigned __int8 *)v8[v9 - 1];
    LODWORD(v29) = v6;
    v11 = *v10;
    if ( (unsigned __int8)v11 > 0x1Cu )
      break;
    if ( (_BYTE)v11 == 22 )
      goto LABEL_52;
    if ( (_BYTE)v11 != 17 )
    {
      v12 = 0;
      goto LABEL_20;
    }
LABEL_18:
    if ( !(_DWORD)v6 )
    {
      v12 = v7;
      goto LABEL_20;
    }
  }
  LOBYTE(a1) = (_BYTE)v11 == 93 || (unsigned __int8)(v11 - 67) <= 1u;
  v12 = (unsigned int)a1;
  if ( (_BYTE)a1 )
  {
    if ( (v10[7] & 0x40) != 0 )
    {
      v19 = (__int64 *)*((_QWORD *)v10 - 1);
    }
    else
    {
      v11 = 32LL * (*((_DWORD *)v10 + 1) & 0x7FFFFFF);
      v19 = (__int64 *)&v10[-v11];
    }
    v15 = *v19;
    if ( v35 )
    {
      v20 = v32;
      v11 = (__int64)&v32[HIDWORD(v33)];
      if ( v32 != (unsigned __int8 **)v11 )
      {
        while ( (unsigned __int8 *)v15 != *v20 )
        {
          if ( (unsigned __int8 **)v11 == ++v20 )
            goto LABEL_31;
        }
        goto LABEL_17;
      }
LABEL_31:
      if ( HIDWORD(v33) < (unsigned int)v33 )
      {
        ++HIDWORD(v33);
        *(_QWORD *)v11 = v15;
        v6 = (unsigned int)v29;
        ++v31;
        goto LABEL_33;
      }
    }
LABEL_38:
    a1 = (unsigned __int8 *)&v31;
    sub_C8CC70((__int64)&v31, v15, v6, v11, (__int64)v8, a6);
    v23 = v22;
    v6 = (unsigned int)v29;
    if ( !v23 )
      goto LABEL_17;
    v21 = (unsigned int)v29 + 1LL;
    if ( v21 > HIDWORD(v29) )
      goto LABEL_40;
    goto LABEL_34;
  }
  if ( (unsigned __int8)(v11 - 54) <= 4u )
  {
    if ( (v10[7] & 0x40) != 0 )
    {
      v13 = (__int64 *)*((_QWORD *)v10 - 1);
    }
    else
    {
      v6 = 32LL * (*((_DWORD *)v10 + 1) & 0x7FFFFFF);
      v13 = (__int64 *)&v10[-v6];
    }
    v14 = *v13;
    v15 = v13[4];
    if ( !v35 )
      goto LABEL_36;
    v16 = v32;
    v17 = HIDWORD(v33);
    v11 = (__int64)&v32[HIDWORD(v33)];
    v6 = (__int64)v32;
    if ( v32 != (unsigned __int8 **)v11 )
    {
      while ( v14 != *(_QWORD *)v6 )
      {
        v6 += 8;
        if ( v11 == v6 )
          goto LABEL_35;
      }
LABEL_12:
      v6 = (__int64)&v16[v17];
      if ( v16 != (unsigned __int8 **)v6 )
      {
        while ( (unsigned __int8 *)v15 != *v16 )
        {
          if ( (unsigned __int8 **)v6 == ++v16 )
            goto LABEL_54;
        }
        LODWORD(v6) = v29;
        goto LABEL_17;
      }
LABEL_54:
      if ( v17 < (unsigned int)v33 )
      {
        HIDWORD(v33) = v17 + 1;
        *(_QWORD *)v6 = v15;
        v6 = (unsigned int)v29;
        ++v31;
LABEL_33:
        v21 = v6 + 1;
        if ( v6 + 1 > (unsigned __int64)HIDWORD(v29) )
        {
LABEL_40:
          a1 = (unsigned __int8 *)&v28;
          sub_C8D5F0((__int64)&v28, v30, v21, 8u, v21, a6);
          v6 = (unsigned int)v29;
        }
LABEL_34:
        v28[v6] = v15;
        LODWORD(v6) = v29 + 1;
        LODWORD(v29) = v29 + 1;
LABEL_17:
        v8 = v28;
        goto LABEL_18;
      }
      goto LABEL_38;
    }
LABEL_35:
    if ( HIDWORD(v33) < (unsigned int)v33 )
    {
      ++HIDWORD(v33);
      *(_QWORD *)v11 = v14;
      ++v31;
    }
    else
    {
LABEL_36:
      a1 = (unsigned __int8 *)&v31;
      sub_C8CC70((__int64)&v31, v14, v6, v11, (__int64)v8, a6);
      if ( !(_BYTE)v6 )
      {
        if ( !v35 )
          goto LABEL_38;
LABEL_60:
        v16 = v32;
        v17 = HIDWORD(v33);
        goto LABEL_12;
      }
    }
    v26 = (unsigned int)v29;
    v11 = HIDWORD(v29);
    v27 = (unsigned int)v29 + 1LL;
    if ( v27 > HIDWORD(v29) )
    {
      a1 = (unsigned __int8 *)&v28;
      sub_C8D5F0((__int64)&v28, v30, v27, 8u, (__int64)v8, a6);
      v26 = (unsigned int)v29;
    }
    v6 = (__int64)v28;
    v28[v26] = v14;
    LODWORD(v29) = v29 + 1;
    if ( !v35 )
      goto LABEL_38;
    goto LABEL_60;
  }
  if ( (_BYTE)v11 != 61 )
    goto LABEL_20;
  a1 = (unsigned __int8 *)*((_QWORD *)v10 - 4);
  v24 = *((_QWORD *)a1 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v24 + 8) - 17 <= 1 )
    v24 = **(_QWORD **)(v24 + 16);
  if ( *(_DWORD *)(v24 + 8) >> 8 == 101 )
  {
LABEL_52:
    v7 = 1;
    goto LABEL_18;
  }
  a1 = sub_98ACB0(a1, 6u);
  if ( *a1 == 22 )
  {
    v25 = sub_B2D680((__int64)a1);
    v12 = v25;
    if ( (_BYTE)v25 )
    {
      LODWORD(v6) = v29;
      v8 = v28;
      v7 = v25;
      goto LABEL_18;
    }
  }
  v8 = v28;
LABEL_20:
  if ( v8 != v30 )
    _libc_free((unsigned __int64)v8);
  if ( !v35 )
    _libc_free((unsigned __int64)v32);
  return v12;
}
