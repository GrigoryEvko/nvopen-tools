// Function: sub_2D65750
// Address: 0x2d65750
//
__int64 __fastcall sub_2D65750(__int64 a1, __int64 a2, _DWORD *a3)
{
  unsigned int v5; // eax
  __int64 v6; // r15
  _BYTE *v7; // rdi
  int v8; // esi
  __int64 (__fastcall *v9)(__int64, int, __int16, __int64, unsigned int); // r13
  unsigned __int8 v10; // al
  __int64 v11; // rsi
  __int64 *v12; // rax
  __int64 *v13; // rdx
  __int64 v14; // rax
  unsigned int v15; // r8d
  __int64 v17; // r15
  __int64 v18; // rdx
  unsigned int v19; // r13d
  __int64 v20; // rdi
  int v21; // eax
  bool v22; // al
  __int64 v23; // rax
  __int64 v24; // rsi
  char v25; // al
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r13
  _BYTE *v29; // rax
  unsigned int v30; // r13d
  __int64 v31; // rax
  unsigned int v32; // esi
  int v33; // eax
  char v34; // [rsp+7h] [rbp-89h]
  int v35; // [rsp+8h] [rbp-88h]
  unsigned __int8 v36; // [rsp+Ch] [rbp-84h]
  unsigned __int8 v37; // [rsp+Ch] [rbp-84h]
  unsigned __int8 v38; // [rsp+Ch] [rbp-84h]
  unsigned __int8 v40; // [rsp+18h] [rbp-78h]
  __int64 v41; // [rsp+18h] [rbp-78h]
  __int64 v42; // [rsp+18h] [rbp-78h]
  __int64 v43; // [rsp+18h] [rbp-78h]
  __int64 v44; // [rsp+28h] [rbp-68h] BYREF
  __int64 v45; // [rsp+30h] [rbp-60h] BYREF
  __int64 v46; // [rsp+38h] [rbp-58h] BYREF
  __int64 *v47; // [rsp+40h] [rbp-50h] BYREF
  __int64 *v48; // [rsp+48h] [rbp-48h]
  __int64 *v49; // [rsp+50h] [rbp-40h]

  v47 = &v44;
  v48 = &v45;
  v49 = &v46;
  v5 = sub_2D65330(&v47, (_BYTE *)a2);
  if ( !(_BYTE)v5 )
  {
    v17 = *(_QWORD *)(a2 - 64);
    if ( *(_BYTE *)v17 <= 0x15u )
      return 0;
    v15 = v5;
    v18 = *(_QWORD *)(a2 - 32);
    if ( (*(_WORD *)(a2 + 2) & 0x3F) == 0x20 )
    {
      v24 = *(_QWORD *)(a2 - 32);
      v42 = *(_QWORD *)(a2 - 32);
      v37 = v5;
      v47 = 0;
      v25 = sub_995B10(&v47, v24);
      v15 = v37;
      if ( !v25 )
        return v15;
      v23 = sub_AD64C0(*(_QWORD *)(v42 + 8), 1, 0);
      goto LABEL_30;
    }
    if ( (*(_WORD *)(a2 + 2) & 0x3F) != 0x21 )
      return v15;
    if ( *(_BYTE *)v18 == 17 )
    {
      v19 = *(_DWORD *)(v18 + 32);
      if ( v19 <= 0x40 )
      {
        v22 = *(_QWORD *)(v18 + 24) == 0;
LABEL_21:
        if ( !v22 )
          return v15;
        goto LABEL_22;
      }
      v36 = v5;
      v20 = v18 + 24;
      v41 = *(_QWORD *)(a2 - 32);
    }
    else
    {
      v28 = *(_QWORD *)(v18 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v28 + 8) - 17 > 1 || *(_BYTE *)v18 > 0x15u )
        return v15;
      v36 = v5;
      v41 = *(_QWORD *)(a2 - 32);
      v29 = sub_AD7630(v41, 0, v18);
      v18 = v41;
      v15 = v36;
      if ( !v29 || *v29 != 17 )
      {
        if ( *(_BYTE *)(v28 + 8) == 17 )
        {
          v35 = *(_DWORD *)(v28 + 32);
          if ( v35 )
          {
            v34 = 0;
            v30 = 0;
            while ( 1 )
            {
              v38 = v15;
              v43 = v18;
              v31 = sub_AD69F0((unsigned __int8 *)v18, v30);
              v15 = v38;
              if ( !v31 )
                break;
              v18 = v43;
              if ( *(_BYTE *)v31 != 13 )
              {
                if ( *(_BYTE *)v31 != 17 )
                  return v15;
                v32 = *(_DWORD *)(v31 + 32);
                if ( v32 <= 0x40 )
                {
                  if ( *(_QWORD *)(v31 + 24) )
                    return v15;
                  v34 = 1;
                }
                else
                {
                  v33 = sub_C444A0(v31 + 24);
                  v15 = v38;
                  if ( v32 != v33 )
                    return v15;
                  v34 = 1;
                  v18 = v43;
                }
              }
              if ( v35 == ++v30 )
              {
                if ( v34 )
                  goto LABEL_22;
                return v15;
              }
            }
          }
        }
        return v15;
      }
      v19 = *((_DWORD *)v29 + 8);
      if ( v19 <= 0x40 )
      {
        if ( *((_QWORD *)v29 + 3) )
          return v15;
LABEL_22:
        v23 = sub_AD62B0(*(_QWORD *)(v18 + 8));
LABEL_30:
        v26 = *(_QWORD *)(v17 + 16);
        if ( !v26 )
          return 0;
        while ( 1 )
        {
          v7 = *(_BYTE **)(v26 + 24);
          if ( *v7 == 42 )
          {
            v27 = *((_QWORD *)v7 - 8);
            if ( v17 == v27 && v27 && v23 == *((_QWORD *)v7 - 4) )
              break;
          }
          v26 = *(_QWORD *)(v26 + 8);
          if ( !v26 )
            return 0;
        }
        v46 = *(_QWORD *)(v26 + 24);
        v8 = 1;
        v6 = *(_QWORD *)(a1 + 16);
        v44 = *((_QWORD *)v7 - 8);
        v45 = *((_QWORD *)v7 - 4);
        v9 = *(__int64 (__fastcall **)(__int64, int, __int16, __int64, unsigned int))(*(_QWORD *)v6 + 1704LL);
        goto LABEL_3;
      }
      v20 = (__int64)(v29 + 24);
    }
    v21 = sub_C444A0(v20);
    v18 = v41;
    v15 = v36;
    v22 = v19 == v21;
    goto LABEL_21;
  }
  v6 = *(_QWORD *)(a1 + 16);
  v7 = (_BYTE *)v46;
  v8 = 2;
  v9 = *(__int64 (__fastcall **)(__int64, int, __int16, __int64, unsigned int))(*(_QWORD *)v6 + 1704LL);
LABEL_3:
  v10 = sub_BD3660((__int64)v7, v8);
  v11 = *(_QWORD *)(a1 + 816);
  v40 = v10;
  v12 = (__int64 *)sub_2D5BAE0(*(_QWORD *)(a1 + 16), v11, *(__int64 **)(v46 + 8), 0);
  if ( v9 != sub_2D56A80 )
  {
    if ( (unsigned __int8)v9(v6, 77, (__int16)v12, (__int64)v13, v40) )
    {
LABEL_7:
      if ( *(_QWORD *)(a2 + 40) == *(_QWORD *)(v46 + 40) || (v14 = *(_QWORD *)(v46 + 16)) != 0 && !*(_QWORD *)(v14 + 8) )
      {
        v15 = sub_2D5EFD0(a1, v46, v44, v45, (_QWORD *)a2, 0x168u);
        if ( (_BYTE)v15 )
        {
          *a3 = 2;
          return v15;
        }
      }
    }
    return 0;
  }
  v47 = v12;
  v48 = v13;
  if ( (_WORD)v12 )
  {
    if ( (unsigned __int16)((_WORD)v12 - 17) > 0xD3u && v40 )
      goto LABEL_7;
    return 0;
  }
  v15 = sub_30070B0(&v47, v11, (unsigned int)v12);
  if ( (_BYTE)v15 || !v40 )
    return 0;
  return v15;
}
