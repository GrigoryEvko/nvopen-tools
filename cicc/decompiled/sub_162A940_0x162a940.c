// Function: sub_162A940
// Address: 0x162a940
//
__int64 __fastcall sub_162A940(__int64 a1, __int64 a2)
{
  int v4; // esi
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // rdx
  __int64 v8; // rdi
  int v9; // r13d
  __int64 v10; // r12
  char v11; // al
  _QWORD *v12; // rdx
  __int64 result; // rax
  int v14; // ecx
  int v15; // eax
  unsigned int v16; // ecx
  __int64 *v17; // rsi
  __int64 v18; // rdi
  int v19; // r11d
  unsigned int v20; // esi
  int v21; // eax
  int v22; // eax
  __int64 v23; // r11
  __int64 v24; // r13
  __int64 v25; // r13
  char v26; // r13
  char v27; // [rsp+Bh] [rbp-D5h]
  int i; // [rsp+3Ch] [rbp-A4h]
  __int64 v29; // [rsp+48h] [rbp-98h] BYREF
  _QWORD *v30; // [rsp+50h] [rbp-90h] BYREF
  __int64 v31; // [rsp+58h] [rbp-88h] BYREF
  __int64 v32; // [rsp+60h] [rbp-80h] BYREF
  int v33; // [rsp+68h] [rbp-78h] BYREF
  __int64 v34; // [rsp+70h] [rbp-70h] BYREF
  __int64 v35; // [rsp+78h] [rbp-68h] BYREF
  __int64 v36; // [rsp+80h] [rbp-60h]
  __int64 v37; // [rsp+88h] [rbp-58h]
  int v38; // [rsp+90h] [rbp-50h]
  int v39; // [rsp+94h] [rbp-4Ch]
  char v40; // [rsp+98h] [rbp-48h]
  int v41; // [rsp+9Ch] [rbp-44h] BYREF
  __int64 v42; // [rsp+A0h] [rbp-40h]

  v29 = a1;
  v4 = *(unsigned __int16 *)(a1 + 2);
  LODWORD(v30) = v4;
  v5 = *(unsigned int *)(a1 + 8);
  v6 = *(_QWORD *)(a1 + 8 * (2 - v5));
  v7 = a1;
  v31 = v6;
  if ( *(_BYTE *)a1 != 15 )
    v7 = *(_QWORD *)(a1 - 8LL * *(unsigned int *)(a1 + 8));
  v32 = v7;
  v33 = *(_DWORD *)(a1 + 24);
  v8 = *(_QWORD *)(a1 + 8 * (1 - v5));
  v34 = *(_QWORD *)(a1 + 8 * (1 - v5));
  v35 = *(_QWORD *)(a1 + 8 * (3 - v5));
  v36 = *(_QWORD *)(a1 + 32);
  v37 = *(_QWORD *)(a1 + 40);
  v38 = *(_DWORD *)(a1 + 48);
  v40 = *(_BYTE *)(a1 + 56);
  if ( v40 )
    v39 = *(_DWORD *)(a1 + 52);
  v9 = *(_DWORD *)(a2 + 24);
  v10 = *(_QWORD *)(a2 + 8);
  v41 = *(_DWORD *)(a1 + 28);
  v42 = *(_QWORD *)(a1 + 8 * (4LL - *(unsigned int *)(a1 + 8)));
  if ( !v9
    || (v6 == 0 || v8 == 0 || v4 != 13 || *(_BYTE *)v8 != 13 || !*(_QWORD *)(v8 + 8 * (7LL - *(unsigned int *)(v8 + 8)))
      ? (v14 = sub_15B4C20((int *)&v30, &v31, &v32, &v33, &v34, &v35, &v41))
      : (v14 = sub_15B2D00(&v31, &v34)),
        v15 = v9 - 1,
        v16 = (v9 - 1) & v14,
        v17 = (__int64 *)(v10 + 8LL * v16),
        v18 = *v17,
        *v17 == -8) )
  {
LABEL_6:
    v11 = sub_15B7AF0(a2, &v29, &v30);
    v12 = v30;
    if ( v11 )
      return v29;
    v20 = *(_DWORD *)(a2 + 24);
    v21 = *(_DWORD *)(a2 + 16);
    ++*(_QWORD *)a2;
    v22 = v21 + 1;
    if ( 4 * v22 >= 3 * v20 )
    {
      v20 *= 2;
    }
    else if ( v20 - *(_DWORD *)(a2 + 20) - v22 > v20 >> 3 )
    {
LABEL_26:
      *(_DWORD *)(a2 + 16) = v22;
      if ( *v12 != -8 )
        --*(_DWORD *)(a2 + 20);
      *v12 = v29;
      return v29;
    }
    sub_15BD0E0(a2, v20);
    sub_15B7AF0(a2, &v29, &v30);
    v12 = v30;
    v22 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_26;
  }
  for ( i = 1; ; ++i )
  {
    if ( v18 == -16 )
      goto LABEL_19;
    v19 = *(unsigned __int16 *)(v18 + 2);
    if ( v34 != 0
      && v31 != 0
      && (_DWORD)v30 == 13
      && *(_BYTE *)v34 == 13
      && *(_QWORD *)(v34 + 8 * (7LL - *(unsigned int *)(v34 + 8)))
      && (_WORD)v19 == 13 )
    {
      v23 = *(unsigned int *)(v18 + 8);
      v24 = *(_QWORD *)(v18 + 8 * (2 - v23));
      if ( v24 && v31 == v24 )
      {
        if ( v34 == *(_QWORD *)(v18 + 8 * (1 - v23)) )
          goto LABEL_46;
        goto LABEL_32;
      }
    }
    else
    {
      if ( (_DWORD)v30 != v19 )
        goto LABEL_19;
      v23 = *(unsigned int *)(v18 + 8);
      v24 = *(_QWORD *)(v18 + 8 * (2 - v23));
    }
    if ( v24 != v31 )
      goto LABEL_19;
LABEL_32:
    v25 = v18;
    if ( *(_BYTE *)v18 != 15 )
      v25 = *(_QWORD *)(v18 - 8 * v23);
    if ( v32 == v25
      && v33 == *(_DWORD *)(v18 + 24)
      && v34 == *(_QWORD *)(v18 + 8 * (1 - v23))
      && v35 == *(_QWORD *)(v18 + 8 * (3 - v23))
      && v36 == *(_QWORD *)(v18 + 32)
      && v38 == *(_DWORD *)(v18 + 48)
      && v37 == *(_QWORD *)(v18 + 40) )
    {
      break;
    }
LABEL_19:
    v16 = v15 & (i + v16);
    v17 = (__int64 *)(v10 + 8LL * v16);
    v18 = *v17;
    if ( *v17 == -8 )
      goto LABEL_6;
  }
  v27 = *(_BYTE *)(v18 + 56);
  if ( v27 )
  {
    if ( v40 )
    {
      if ( *(_DWORD *)(v18 + 52) != v39 )
        goto LABEL_19;
      goto LABEL_44;
    }
    v26 = 0;
  }
  else
  {
    v26 = v40;
  }
  if ( v27 != v26 )
    goto LABEL_19;
LABEL_44:
  if ( v41 != *(_DWORD *)(v18 + 28) || v42 != *(_QWORD *)(v18 + 8 * (4 - v23)) )
    goto LABEL_19;
LABEL_46:
  if ( (__int64 *)(*(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24)) == v17 )
    goto LABEL_6;
  result = *v17;
  if ( !*v17 )
    goto LABEL_6;
  return result;
}
