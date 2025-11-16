// Function: sub_11451F0
// Address: 0x11451f0
//
__int64 __fastcall sub_11451F0(__int64 a1, __int64 *a2)
{
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v12; // rcx
  unsigned __int64 v13; // rax
  int *v14; // rdx
  int v15; // eax
  __int16 v16; // r14
  __int64 v17; // rcx
  __int64 v18; // rax
  int *v19; // rdx
  char v20; // al
  _BYTE *v21; // rbx
  __int64 v22; // rax
  unsigned __int64 v23; // rax
  int *v24; // rdx
  char v25; // al
  _BYTE *v26; // r14
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  int *v30; // rdx
  const char *v31; // rax
  __int64 v32; // rsi
  _QWORD *v33; // rdx
  __int64 v34; // r12
  __int64 v35; // rax
  __int64 v36; // r9
  char v37; // al
  _BYTE *v38; // rbx
  _BYTE *v39; // rbx
  __int64 v40; // rax
  char v41; // al
  _BYTE *v42; // r14
  __int64 v43; // rax
  __int64 v44; // [rsp+38h] [rbp-88h] BYREF
  __int64 v45; // [rsp+40h] [rbp-80h] BYREF
  int v46; // [rsp+48h] [rbp-78h] BYREF
  char v47; // [rsp+4Ch] [rbp-74h]
  int *v48; // [rsp+50h] [rbp-70h] BYREF
  _QWORD *v49; // [rsp+58h] [rbp-68h] BYREF
  __int64 *v50; // [rsp+60h] [rbp-60h] BYREF
  __int64 *v51; // [rsp+68h] [rbp-58h]
  _QWORD *v52; // [rsp+70h] [rbp-50h] BYREF
  __int64 *v53; // [rsp+78h] [rbp-48h]
  _QWORD *v54; // [rsp+80h] [rbp-40h] BYREF
  __int64 *v55; // [rsp+88h] [rbp-38h]

  v3 = *(_QWORD *)(a1 - 64);
  v48 = &v46;
  v49 = 0;
  v50 = &v45;
  v51 = &v44;
  v4 = *(_QWORD *)(v3 + 16);
  v46 = 42;
  v47 = 0;
  if ( !v4
    || *(_QWORD *)(v4 + 8)
    || *(_BYTE *)v3 != 54
    || !(unsigned __int8)sub_993A50(&v49, *(_QWORD *)(v3 - 64))
    || (v12 = *(_QWORD *)(v3 - 32)) == 0 )
  {
    v5 = *(_QWORD *)(a1 - 32);
LABEL_3:
    v6 = *(_QWORD *)(v5 + 16);
    if ( v6
      && !*(_QWORD *)(v6 + 8)
      && *(_BYTE *)v5 == 54
      && (unsigned __int8)sub_993A50(&v49, *(_QWORD *)(v5 - 64))
      && (v17 = *(_QWORD *)(v5 - 32)) != 0 )
    {
      *v50 = v17;
      v7 = *(_QWORD *)(a1 - 64);
      if ( v7 )
      {
        *v51 = v7;
        if ( v48 )
        {
          v18 = sub_B53960(a1);
          v19 = v48;
          *v48 = v18;
          *((_BYTE *)v19 + 4) = BYTE4(v18);
        }
LABEL_24:
        v15 = v46;
        if ( v46 == 34 )
          goto LABEL_16;
LABEL_25:
        if ( v15 != 37 )
          return 0;
LABEL_49:
        v16 = 33;
        goto LABEL_50;
      }
    }
    else
    {
      v7 = *(_QWORD *)(a1 - 64);
    }
    v48 = &v46;
    v49 = 0;
    v50 = 0;
    v51 = &v45;
    v52 = 0;
    v53 = &v45;
    v54 = 0;
    v55 = &v44;
    v8 = *(_QWORD *)(v7 + 16);
    if ( !v8 || *(_QWORD *)(v8 + 8) )
      goto LABEL_6;
    v20 = *(_BYTE *)v7;
    if ( *(_BYTE *)v7 != 59 )
      goto LABEL_29;
    v37 = sub_995B10(&v49, *(_QWORD *)(v7 - 64));
    v38 = *(_BYTE **)(v7 - 32);
    if ( v37 && *v38 == 54 )
    {
      if ( (unsigned __int8)sub_995B10(&v50, *((_QWORD *)v38 - 8)) )
      {
        v40 = *((_QWORD *)v38 - 4);
        if ( v40 )
          goto LABEL_57;
      }
      v38 = *(_BYTE **)(v7 - 32);
    }
    if ( !(unsigned __int8)sub_995B10(&v49, (__int64)v38) )
      goto LABEL_63;
    v39 = *(_BYTE **)(v7 - 64);
    if ( *v39 != 54 )
      goto LABEL_6;
    if ( !(unsigned __int8)sub_995B10(&v50, *((_QWORD *)v39 - 8)) || (v40 = *((_QWORD *)v39 - 4)) == 0 )
    {
LABEL_63:
      v20 = *(_BYTE *)v7;
LABEL_29:
      if ( v20 == 42 )
      {
        v21 = *(_BYTE **)(v7 - 64);
        if ( *v21 == 54 )
        {
          if ( (unsigned __int8)sub_993A50(&v52, *((_QWORD *)v21 - 8)) )
          {
            v22 = *((_QWORD *)v21 - 4);
            if ( v22 )
            {
              *v53 = v22;
              if ( (unsigned __int8)sub_995B10(&v54, *(_QWORD *)(v7 - 32)) )
              {
LABEL_34:
                v9 = *(_QWORD *)(a1 - 32);
                if ( v9 )
                {
                  *v55 = v9;
                  if ( v48 )
                  {
                    v23 = sub_B53900(a1);
                    v24 = v48;
                    *v48 = v23;
                    *((_BYTE *)v24 + 4) = BYTE4(v23);
                  }
                  goto LABEL_47;
                }
LABEL_7:
                v10 = *(_QWORD *)(v9 + 16);
                if ( !v10 || *(_QWORD *)(v10 + 8) )
                  return 0;
                v25 = *(_BYTE *)v9;
                if ( *(_BYTE *)v9 != 59 )
                {
LABEL_39:
                  if ( v25 != 42 )
                    return 0;
                  v26 = *(_BYTE **)(v9 - 64);
                  if ( *v26 != 54 )
                    return 0;
                  if ( !(unsigned __int8)sub_993A50(&v52, *((_QWORD *)v26 - 8)) )
                    return 0;
                  v27 = *((_QWORD *)v26 - 4);
                  if ( !v27 )
                    return 0;
                  *v53 = v27;
                  if ( !(unsigned __int8)sub_995B10(&v54, *(_QWORD *)(v9 - 32)) )
                    return 0;
                  goto LABEL_44;
                }
                v41 = sub_995B10(&v49, *(_QWORD *)(v9 - 64));
                v42 = *(_BYTE **)(v9 - 32);
                if ( v41 && *v42 == 54 )
                {
                  if ( (unsigned __int8)sub_995B10(&v50, *((_QWORD *)v42 - 8)) )
                  {
                    v43 = *((_QWORD *)v42 - 4);
                    if ( v43 )
                    {
                      *v51 = v43;
                      goto LABEL_44;
                    }
                  }
                  v42 = *(_BYTE **)(v9 - 32);
                }
                if ( !(unsigned __int8)sub_995B10(&v49, (__int64)v42)
                  || !(unsigned __int8)sub_100AC40(&v50, 25, *(unsigned __int8 **)(v9 - 64)) )
                {
                  v25 = *(_BYTE *)v9;
                  goto LABEL_39;
                }
LABEL_44:
                v28 = *(_QWORD *)(a1 - 64);
                if ( !v28 )
                  return 0;
                *v55 = v28;
                if ( v48 )
                {
                  v29 = sub_B53960(a1);
                  v30 = v48;
                  *v48 = v29;
                  *((_BYTE *)v30 + 4) = BYTE4(v29);
                }
LABEL_47:
                if ( v46 == 35 )
                  goto LABEL_16;
                if ( v46 == 36 )
                  goto LABEL_49;
                return 0;
              }
            }
          }
        }
      }
LABEL_6:
      v9 = *(_QWORD *)(a1 - 32);
      goto LABEL_7;
    }
LABEL_57:
    *v51 = v40;
    goto LABEL_34;
  }
  *v50 = v12;
  v5 = *(_QWORD *)(a1 - 32);
  if ( !v5 )
    goto LABEL_3;
  *v51 = v5;
  if ( !v48 )
    goto LABEL_24;
  v13 = sub_B53900(a1);
  v14 = v48;
  *v48 = v13;
  *((_BYTE *)v14 + 4) = BYTE4(v13);
  v15 = v46;
  if ( v46 != 34 )
    goto LABEL_25;
LABEL_16:
  v16 = 32;
LABEL_50:
  v31 = sub_BD5D20(v44);
  v32 = v44;
  v49 = v33;
  LOWORD(v52) = 773;
  v48 = (int *)v31;
  v50 = (__int64 *)".highbits";
  v34 = sub_F94560(a2, v44, v45, (__int64)&v48, 0);
  v35 = sub_AD6530(*(_QWORD *)(v34 + 8), v32);
  LOWORD(v52) = 257;
  return sub_B52500(53, v16, v34, v35, (__int64)&v48, v36, 0, 0);
}
