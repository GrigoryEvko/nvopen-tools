// Function: sub_F12440
// Address: 0xf12440
//
unsigned __int8 *__fastcall sub_F12440(__int64 a1, unsigned __int8 *a2)
{
  char *v2; // r14
  unsigned __int8 *v3; // r12
  unsigned __int8 v4; // bl
  unsigned __int8 v5; // al
  unsigned __int8 *v6; // r12
  __int64 v8; // r13
  int v9; // r8d
  __int64 *v10; // r12
  __int64 v11; // rbx
  __int64 v12; // rdx
  char v13; // al
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // r15
  __int64 v17; // r13
  unsigned __int8 *v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 *v25; // r14
  __int64 v26; // rax
  __int64 v27; // r14
  __int64 v28; // r13
  unsigned __int8 *v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // [rsp+0h] [rbp-B0h]
  __int64 v37; // [rsp+8h] [rbp-A8h]
  int v38; // [rsp+1Ch] [rbp-94h] BYREF
  char *v39; // [rsp+20h] [rbp-90h] BYREF
  char *v40; // [rsp+28h] [rbp-88h] BYREF
  _QWORD v41[4]; // [rsp+30h] [rbp-80h] BYREF
  _QWORD *v42; // [rsp+50h] [rbp-60h] BYREF
  __int64 v43; // [rsp+58h] [rbp-58h]
  __int16 v44; // [rsp+70h] [rbp-40h]

  v2 = (char *)*((_QWORD *)a2 - 8);
  v3 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
  v38 = *a2 - 29;
  v39 = (char *)v3;
  v4 = *v2;
  if ( (unsigned __int8)*v2 > 0x1Cu && (v4 == 68 || v4 == 69) && (v8 = *((_QWORD *)v2 - 4)) != 0 )
  {
    v9 = sub_BCB060(*(_QWORD *)(v8 + 8));
    v5 = *v3;
    if ( v9 == 1 && v5 == 86 )
    {
      if ( (v3[7] & 0x40) != 0 )
      {
        v10 = (__int64 *)*((_QWORD *)v3 - 1);
        v11 = *v10;
        if ( !*v10 )
          return 0;
      }
      else
      {
        v10 = (__int64 *)&v3[-32 * (*((_DWORD *)v3 + 1) & 0x7FFFFFF)];
        v11 = *v10;
        if ( !*v10 )
          return 0;
      }
      v12 = v10[4];
      if ( v12 )
      {
        v37 = v10[8];
        if ( v37 )
        {
          v40 = v2;
          goto LABEL_16;
        }
      }
      return 0;
    }
  }
  else
  {
    v5 = *v3;
  }
  if ( v5 <= 0x1Cu || v5 != 68 && v5 != 69 )
    return 0;
  v8 = *((_QWORD *)v3 - 4);
  if ( !v8 || (unsigned int)sub_BCB060(*(_QWORD *)(v8 + 8)) != 1 || v4 != 86 )
    return 0;
  if ( (v2[7] & 0x40) != 0 )
  {
    v25 = (__int64 *)*((_QWORD *)v2 - 1);
    v11 = *v25;
    if ( !*v25 )
      return 0;
  }
  else
  {
    v25 = (__int64 *)&v2[-32 * (*((_DWORD *)v2 + 1) & 0x7FFFFFF)];
    v11 = *v25;
    if ( !*v25 )
      return 0;
  }
  v12 = v25[4];
  if ( !v12 )
    return 0;
  v37 = v25[8];
  if ( !v37 )
    return 0;
  v40 = (char *)v3;
LABEL_16:
  v41[2] = a1;
  v41[0] = &v40;
  v41[1] = &v39;
  v41[3] = &v38;
  if ( v11 == v8 )
  {
    v26 = sub_F0AE50((__int64)v41, 0, v12);
    v44 = 257;
    v27 = v26;
    v28 = sub_F0AE50((__int64)v41, 1, v37);
    v29 = (unsigned __int8 *)sub_BD2C40(72, 3u);
    v6 = v29;
    if ( !v29 )
      return v6;
    sub_B44260((__int64)v29, *(_QWORD *)(v27 + 8), 57, 3u, 0, 0);
    if ( *((_QWORD *)v6 - 12) )
    {
      v30 = *((_QWORD *)v6 - 11);
      **((_QWORD **)v6 - 10) = v30;
      if ( v30 )
        *(_QWORD *)(v30 + 16) = *((_QWORD *)v6 - 10);
    }
    *((_QWORD *)v6 - 12) = v11;
    v31 = *(_QWORD *)(v11 + 16);
    *((_QWORD *)v6 - 11) = v31;
    if ( v31 )
      *(_QWORD *)(v31 + 16) = v6 - 88;
    *((_QWORD *)v6 - 10) = v11 + 16;
    *(_QWORD *)(v11 + 16) = v6 - 96;
    if ( *((_QWORD *)v6 - 8) )
    {
      v32 = *((_QWORD *)v6 - 7);
      **((_QWORD **)v6 - 6) = v32;
      if ( v32 )
        *(_QWORD *)(v32 + 16) = *((_QWORD *)v6 - 6);
    }
    *((_QWORD *)v6 - 8) = v27;
    v33 = *(_QWORD *)(v27 + 16);
    *((_QWORD *)v6 - 7) = v33;
    if ( v33 )
      *(_QWORD *)(v33 + 16) = v6 - 56;
    *((_QWORD *)v6 - 6) = v27 + 16;
    *(_QWORD *)(v27 + 16) = v6 - 64;
    if ( *((_QWORD *)v6 - 4) )
    {
      v34 = *((_QWORD *)v6 - 3);
      **((_QWORD **)v6 - 2) = v34;
      if ( v34 )
        *(_QWORD *)(v34 + 16) = *((_QWORD *)v6 - 2);
    }
    *((_QWORD *)v6 - 4) = v28;
    if ( v28 )
    {
      v35 = *(_QWORD *)(v28 + 16);
      *((_QWORD *)v6 - 3) = v35;
      if ( v35 )
        *(_QWORD *)(v35 + 16) = v6 - 24;
      *((_QWORD *)v6 - 2) = v28 + 16;
      *(_QWORD *)(v28 + 16) = v6 - 32;
    }
    goto LABEL_40;
  }
  v42 = 0;
  v43 = v11;
  if ( *(_BYTE *)v8 != 59 )
    return 0;
  v36 = v12;
  v13 = sub_995B10(&v42, *(_QWORD *)(v8 - 64));
  v14 = *(_QWORD *)(v8 - 32);
  v15 = v36;
  if ( !v13 || v14 != v43 )
  {
    if ( !(unsigned __int8)sub_995B10(&v42, v14) )
      return 0;
    v15 = v36;
    if ( *(_QWORD *)(v8 - 64) != v43 )
      return 0;
  }
  v16 = sub_F0AE50((__int64)v41, 1, v15);
  v44 = 257;
  v17 = sub_F0AE50((__int64)v41, 0, v37);
  v18 = (unsigned __int8 *)sub_BD2C40(72, 3u);
  v6 = v18;
  if ( v18 )
  {
    sub_B44260((__int64)v18, *(_QWORD *)(v16 + 8), 57, 3u, 0, 0);
    if ( *((_QWORD *)v6 - 12) )
    {
      v19 = *((_QWORD *)v6 - 11);
      **((_QWORD **)v6 - 10) = v19;
      if ( v19 )
        *(_QWORD *)(v19 + 16) = *((_QWORD *)v6 - 10);
    }
    *((_QWORD *)v6 - 12) = v11;
    v20 = *(_QWORD *)(v11 + 16);
    *((_QWORD *)v6 - 11) = v20;
    if ( v20 )
      *(_QWORD *)(v20 + 16) = v6 - 88;
    *((_QWORD *)v6 - 10) = v11 + 16;
    *(_QWORD *)(v11 + 16) = v6 - 96;
    if ( *((_QWORD *)v6 - 8) )
    {
      v21 = *((_QWORD *)v6 - 7);
      **((_QWORD **)v6 - 6) = v21;
      if ( v21 )
        *(_QWORD *)(v21 + 16) = *((_QWORD *)v6 - 6);
    }
    *((_QWORD *)v6 - 8) = v16;
    v22 = *(_QWORD *)(v16 + 16);
    *((_QWORD *)v6 - 7) = v22;
    if ( v22 )
      *(_QWORD *)(v22 + 16) = v6 - 56;
    *((_QWORD *)v6 - 6) = v16 + 16;
    *(_QWORD *)(v16 + 16) = v6 - 64;
    if ( *((_QWORD *)v6 - 4) )
    {
      v23 = *((_QWORD *)v6 - 3);
      **((_QWORD **)v6 - 2) = v23;
      if ( v23 )
        *(_QWORD *)(v23 + 16) = *((_QWORD *)v6 - 2);
    }
    *((_QWORD *)v6 - 4) = v17;
    if ( v17 )
    {
      v24 = *(_QWORD *)(v17 + 16);
      *((_QWORD *)v6 - 3) = v24;
      if ( v24 )
        *(_QWORD *)(v24 + 16) = v6 - 24;
      *((_QWORD *)v6 - 2) = v17 + 16;
      *(_QWORD *)(v17 + 16) = v6 - 32;
    }
LABEL_40:
    sub_BD6B50(v6, (const char **)&v42);
  }
  return v6;
}
