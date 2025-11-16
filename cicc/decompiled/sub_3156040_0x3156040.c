// Function: sub_3156040
// Address: 0x3156040
//
__int64 *__fastcall sub_3156040(__int64 *a1, __int64 a2, _QWORD *a3)
{
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  int v10; // edx
  __int64 v11; // rcx
  __int64 *v12; // rdx
  __int64 v13; // rax
  _QWORD *v14; // rax
  const char **v15; // rsi
  char v16; // dl
  char v17; // [rsp+8h] [rbp-208h]
  const char *v18; // [rsp+10h] [rbp-200h] BYREF
  __int64 v19; // [rsp+18h] [rbp-1F8h] BYREF
  __int64 *v20; // [rsp+20h] [rbp-1F0h]
  const char *v21; // [rsp+28h] [rbp-1E8h]
  unsigned __int64 v22[2]; // [rsp+30h] [rbp-1E0h] BYREF
  _BYTE v23[136]; // [rsp+40h] [rbp-1D0h] BYREF
  int v24; // [rsp+C8h] [rbp-148h] BYREF
  _QWORD *v25; // [rsp+D0h] [rbp-140h]
  int *v26; // [rsp+D8h] [rbp-138h]
  int *v27; // [rsp+E0h] [rbp-130h]
  __int64 v28; // [rsp+E8h] [rbp-128h]
  __int64 v29; // [rsp+F0h] [rbp-120h] BYREF
  __int64 v30; // [rsp+F8h] [rbp-118h]
  __int64 *v31; // [rsp+100h] [rbp-110h]
  const char *v32; // [rsp+108h] [rbp-108h]
  char *v33; // [rsp+110h] [rbp-100h] BYREF
  int v34; // [rsp+118h] [rbp-F8h]
  _BYTE v35[136]; // [rsp+120h] [rbp-F0h] BYREF
  int v36; // [rsp+1A8h] [rbp-68h] BYREF
  _QWORD *v37; // [rsp+1B0h] [rbp-60h]
  int *v38; // [rsp+1B8h] [rbp-58h]
  int *v39; // [rsp+1C0h] [rbp-50h]
  __int64 v40; // [rsp+1C8h] [rbp-48h]
  char v41; // [rsp+1D0h] [rbp-40h]

  sub_3154320(&v29, a2, 9);
  if ( (v29 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v29 & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  while ( 1 )
  {
    if ( !sub_3154C60(a2, 10, v5, v6) )
    {
      *a1 = 1;
      return a1;
    }
    sub_31552E0((__int64)&v29, a2, 10);
    v10 = v41 & 1;
    v11 = (unsigned int)(2 * v10);
    v41 = (2 * v10) | v41 & 0xFD;
    if ( (_BYTE)v10 )
    {
      *a1 = v29 | 1;
      return a1;
    }
    v12 = v31;
    v18 = v32;
    v13 = v30;
    v20 = v31;
    v19 = v30;
    if ( v31 )
    {
      *v31 = (__int64)&v19;
      v13 = v30;
    }
    if ( v13 )
    {
      v12 = &v19;
      *(_QWORD *)(v13 + 8) = &v19;
    }
    v31 = 0;
    v30 = 0;
    v21 = v32;
    v22[1] = 0x1000000000LL;
    v22[0] = (unsigned __int64)v23;
    if ( v34 )
    {
      sub_3153680((__int64)v22, &v33, (__int64)v12, v11, v8, v9);
      v14 = v37;
      if ( v37 )
      {
LABEL_17:
        v25 = v14;
        v24 = v36;
        v26 = v38;
        v27 = v39;
        v14[1] = &v24;
        v37 = 0;
        v28 = v40;
        v38 = &v36;
        v39 = &v36;
        v40 = 0;
        goto LABEL_18;
      }
    }
    else
    {
      v14 = v37;
      if ( v37 )
        goto LABEL_17;
    }
    v24 = 0;
    v25 = 0;
    v26 = &v24;
    v27 = &v24;
    v28 = 0;
LABEL_18:
    v15 = &v18;
    sub_3154710(a3, (__int64 *)&v18);
    v17 = v16;
    sub_31541A0(v25);
    if ( (_BYTE *)v22[0] != v23 )
      _libc_free(v22[0]);
    if ( v20 )
    {
      v5 = v19;
      *v20 = v19;
    }
    if ( v19 )
    {
      v5 = (__int64)v20;
      *(_QWORD *)(v19 + 8) = v20;
    }
    if ( !v17 )
      break;
    if ( (v41 & 2) != 0 )
      goto LABEL_33;
    if ( (v41 & 1) != 0 )
    {
      if ( v29 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v29 + 8LL))(v29);
    }
    else
    {
      sub_31541A0(v37);
      if ( v33 != v35 )
        _libc_free((unsigned __int64)v33);
      if ( v31 )
      {
        v5 = v30;
        *v31 = v30;
      }
      if ( v30 )
      {
        v5 = (__int64)v31;
        *(_QWORD *)(v30 + 8) = v31;
      }
    }
  }
  v15 = (const char **)a2;
  v18 = "Duplicate roots";
  LOWORD(v22[0]) = 259;
  sub_31542E0(a1, a2, (void **)&v18);
  if ( (v41 & 2) != 0 )
LABEL_33:
    sub_31551A0(&v29, (__int64)v15);
  if ( (v41 & 1) != 0 )
  {
    if ( v29 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v29 + 8LL))(v29);
  }
  else
  {
    sub_31541A0(v37);
    if ( v33 != v35 )
      _libc_free((unsigned __int64)v33);
    if ( v31 )
      *v31 = v30;
    if ( v30 )
      *(_QWORD *)(v30 + 8) = v31;
  }
  return a1;
}
