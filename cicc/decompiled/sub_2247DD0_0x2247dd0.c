// Function: sub_2247DD0
// Address: 0x2247dd0
//
__int64 *__fastcall sub_2247DD0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        _DWORD *a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        _DWORD *a10)
{
  __int64 *v10; // r12
  void *v11; // rsp
  bool v12; // al
  __int64 v13; // r9
  int v14; // eax
  unsigned __int64 v15; // rbx
  unsigned __int64 v16; // r14
  _DWORD **v17; // r12
  __int64 *v18; // rdi
  char v19; // r15
  char v20; // dl
  char v21; // al
  char v22; // dl
  int v23; // r11d
  unsigned __int64 v24; // rax
  __int64 v25; // rdi
  unsigned __int64 *v26; // rdx
  int *v27; // rsi
  unsigned __int64 v28; // rax
  int *v29; // rax
  int v30; // esi
  int *v32; // rax
  int v33; // eax
  int *v34; // rax
  int v35; // eax
  int v36; // eax
  __int64 v37; // r15
  int v38; // eax
  int v39; // edx
  __int64 v40; // rax
  int v41; // eax
  void *v42; // rsp
  size_t v43; // rax
  __int64 v44; // rax
  int v45; // eax
  __int64 v46; // rax
  _DWORD *v47; // [rsp+0h] [rbp-60h] BYREF
  int v48; // [rsp+Ch] [rbp-54h]
  __int64 *v49; // [rsp+10h] [rbp-50h] BYREF
  __int64 v50; // [rsp+18h] [rbp-48h]
  __int64 *v51; // [rsp+20h] [rbp-40h] BYREF
  __int64 v52; // [rsp+28h] [rbp-38h]
  __int64 v53; // [rsp+70h] [rbp+10h]
  __int64 v54; // [rsp+70h] [rbp+10h]
  __int64 v55; // [rsp+70h] [rbp+10h]
  __int64 v56; // [rsp+70h] [rbp+10h]
  __int64 v57; // [rsp+70h] [rbp+10h]
  __int64 v58; // [rsp+70h] [rbp+10h]
  __int64 v59; // [rsp+70h] [rbp+10h]

  v47 = a6;
  v51 = a2;
  v52 = a3;
  v49 = a4;
  v50 = a5;
  v10 = (__int64 *)sub_2243120((_QWORD *)(a9 + 208), (__int64)a2);
  v11 = alloca(8 * a8 + 8);
  v12 = sub_2247850((__int64)&v51, (__int64)&v49);
  v13 = a7;
  if ( v12 )
  {
    v14 = v52;
    v15 = 0;
    v16 = 0;
    v17 = 0;
  }
  else
  {
    v37 = 0;
    v16 = 0;
    v38 = sub_2247910((__int64)&v51);
    v13 = a7;
    v39 = v38;
    v15 = 2 * a8;
    if ( 2 * a8 )
    {
      do
      {
        while ( **(_DWORD **)(v13 + 8 * v37) != v39 )
        {
          v40 = *v10;
          v54 = v13;
          v48 = v39;
          v41 = (*(__int64 (__fastcall **)(__int64 *))(v40 + 48))(v10);
          v39 = v48;
          v13 = v54;
          if ( v48 == v41 )
            break;
          if ( v15 == ++v37 )
            goto LABEL_50;
        }
        *((_DWORD *)&v47 + v16++) = v37++;
      }
      while ( v15 != v37 );
LABEL_50:
      if ( v16 )
      {
        v55 = v13;
        v15 = 0;
        sub_2240940(v51);
        LODWORD(v52) = -1;
        v13 = v55;
        v42 = alloca(8 * v16 + 8);
        v17 = &v47;
        do
        {
          v56 = v13;
          v43 = wcslen(*(const wchar_t **)(v13 + 8LL * *((int *)&v47 + v15)));
          v13 = v56;
          (&v47)[v15++] = (_DWORD *)v43;
        }
        while ( v16 != v15 );
        v14 = -1;
        v16 = 1;
      }
      else
      {
        v14 = v52;
        v15 = 0;
        v17 = 0;
      }
    }
    else
    {
      v14 = v52;
      v17 = 0;
    }
  }
  v18 = v51;
  while ( 2 )
  {
    v19 = v14 == -1;
    v20 = v19 & (v18 != 0);
    if ( v20 )
    {
      v32 = (int *)v18[2];
      if ( (unsigned __int64)v32 >= v18[3] )
      {
        v46 = *v18;
        v59 = v13;
        LOBYTE(v48) = v19 & (v18 != 0);
        v33 = (*(__int64 (__fastcall **)(__int64 *))(v46 + 72))(v18);
        v13 = v59;
        v20 = v48;
      }
      else
      {
        v33 = *v32;
      }
      v19 = 0;
      if ( v33 == -1 )
      {
        v51 = 0;
        v19 = v20;
      }
    }
    v21 = (_DWORD)v50 == -1;
    v22 = v21 & (v49 != 0);
    if ( !v22 )
      goto LABEL_6;
    v29 = (int *)v49[2];
    if ( (unsigned __int64)v29 >= v49[3] )
    {
      v44 = *v49;
      v58 = v13;
      LOBYTE(v48) = v22;
      v45 = (*(__int64 (**)(void))(v44 + 72))();
      v13 = v58;
      v22 = v48;
      v30 = v45;
    }
    else
    {
      v30 = *v29;
    }
    v21 = 0;
    if ( v30 == -1 )
    {
      v49 = 0;
      if ( v19 == v22 )
        break;
    }
    else
    {
LABEL_6:
      if ( v19 == v21 )
        break;
    }
    v23 = v52;
    if ( (_DWORD)v52 == -1 && v51 )
    {
      v34 = (int *)v51[2];
      if ( (unsigned __int64)v34 >= v51[3] )
      {
        v57 = v13;
        v35 = (*(__int64 (**)(void))(*v51 + 72))();
        v13 = v57;
        v23 = v35;
      }
      else
      {
        v23 = *v34;
        v35 = *v34;
      }
      if ( v35 == -1 )
        v51 = 0;
    }
    if ( !v15 )
    {
LABEL_27:
      *a10 |= 4u;
      return v51;
    }
    v24 = 0;
    v25 = 0;
    do
    {
      while ( 1 )
      {
        v26 = (unsigned __int64 *)&v17[v24];
        if ( *v26 > v16 )
          break;
        ++v25;
        ++v24;
LABEL_12:
        if ( v24 >= v15 )
          goto LABEL_16;
      }
      v27 = (int *)&v47 + v24;
      if ( *(_DWORD *)(*(_QWORD *)(v13 + 8LL * *v27) + 4 * v16) == v23 )
      {
        ++v24;
        goto LABEL_12;
      }
      --v15;
      *v27 = *((_DWORD *)&v47 + v15);
      *v26 = (unsigned __int64)v17[v15];
    }
    while ( v24 < v15 );
LABEL_16:
    if ( v15 != v25 )
    {
      v18 = v51;
      v28 = v51[2];
      if ( v28 >= v51[3] )
      {
        v53 = v13;
        (*(void (__fastcall **)(__int64 *))(*v51 + 80))(v51);
        v18 = v51;
        v13 = v53;
      }
      else
      {
        v51[2] = v28 + 4;
      }
      LODWORD(v52) = -1;
      ++v16;
      v14 = -1;
      continue;
    }
    break;
  }
  if ( v15 != 1 )
  {
    if ( v15 == 2 && (*v17 == (_DWORD *)v16 || v17[1] == (_DWORD *)v16) )
      goto LABEL_41;
    goto LABEL_27;
  }
  if ( *v17 != (_DWORD *)v16 )
    goto LABEL_27;
LABEL_41:
  v36 = (int)v47;
  if ( (int)v47 >= (int)a8 )
    v36 = (_DWORD)v47 - a8;
  *v47 = v36;
  return v51;
}
