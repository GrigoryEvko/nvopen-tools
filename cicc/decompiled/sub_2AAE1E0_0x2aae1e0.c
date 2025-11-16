// Function: sub_2AAE1E0
// Address: 0x2aae1e0
//
unsigned __int8 *__fastcall sub_2AAE1E0(unsigned int ***a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // r14d
  bool v6; // al
  __int64 v8; // r14
  __int64 v9; // rdx
  unsigned int v10; // r14d
  __int64 v11; // r14
  __int64 v12; // rdx
  _BYTE *v13; // rax
  unsigned int v14; // r14d
  bool v15; // al
  __int64 v16; // rdx
  unsigned int **v17; // r14
  __int64 v18; // rax
  _BYTE *v19; // rax
  unsigned int v20; // r14d
  unsigned int **v21; // rdi
  bool v22; // r14
  unsigned int v23; // r15d
  __int64 v24; // rax
  unsigned int v25; // r14d
  bool v26; // r14
  unsigned int v27; // r15d
  __int64 v28; // rax
  unsigned int v29; // r14d
  int v30; // [rsp+Ch] [rbp-74h]
  int v31; // [rsp+Ch] [rbp-74h]
  __int64 v32; // [rsp+18h] [rbp-68h]
  __int64 v33[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v34; // [rsp+40h] [rbp-40h]

  if ( *(_BYTE *)a2 == 17 )
  {
    v5 = *(_DWORD *)(a2 + 32);
    if ( v5 <= 0x40 )
      v6 = *(_QWORD *)(a2 + 24) == 1;
    else
      v6 = v5 - 1 == (unsigned int)sub_C444A0(a2 + 24);
  }
  else
  {
    v8 = *(_QWORD *)(a2 + 8);
    v9 = (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17;
    if ( (unsigned int)v9 > 1 || *(_BYTE *)a2 > 0x15u )
      goto LABEL_8;
    v19 = sub_AD7630(a2, 0, v9);
    if ( !v19 || *v19 != 17 )
    {
      if ( *(_BYTE *)(v8 + 8) == 17 )
      {
        v30 = *(_DWORD *)(v8 + 32);
        if ( v30 )
        {
          v22 = 0;
          v23 = 0;
          while ( 1 )
          {
            v24 = sub_AD69F0((unsigned __int8 *)a2, v23);
            if ( !v24 )
              break;
            if ( *(_BYTE *)v24 != 13 )
            {
              if ( *(_BYTE *)v24 != 17 )
                break;
              v25 = *(_DWORD *)(v24 + 32);
              v22 = v25 <= 0x40 ? *(_QWORD *)(v24 + 24) == 1 : v25 - 1 == (unsigned int)sub_C444A0(v24 + 24);
              if ( !v22 )
                break;
            }
            if ( v30 == ++v23 )
            {
              if ( v22 )
                return (unsigned __int8 *)a3;
              goto LABEL_8;
            }
          }
        }
      }
      goto LABEL_8;
    }
    v20 = *((_DWORD *)v19 + 8);
    if ( v20 <= 0x40 )
      v6 = *((_QWORD *)v19 + 3) == 1;
    else
      v6 = v20 - 1 == (unsigned int)sub_C444A0((__int64)(v19 + 24));
  }
  if ( v6 )
    return (unsigned __int8 *)a3;
LABEL_8:
  if ( *(_BYTE *)a3 == 17 )
  {
    v10 = *(_DWORD *)(a3 + 32);
    if ( v10 <= 0x40 )
    {
      if ( *(_QWORD *)(a3 + 24) == 1 )
        return (unsigned __int8 *)a2;
    }
    else if ( (unsigned int)sub_C444A0(a3 + 24) == v10 - 1 )
    {
      return (unsigned __int8 *)a2;
    }
  }
  else
  {
    v11 = *(_QWORD *)(a3 + 8);
    v12 = (unsigned int)*(unsigned __int8 *)(v11 + 8) - 17;
    if ( (unsigned int)v12 <= 1 && *(_BYTE *)a3 <= 0x15u )
    {
      v13 = sub_AD7630(a3, 0, v12);
      if ( v13 && *v13 == 17 )
      {
        v14 = *((_DWORD *)v13 + 8);
        if ( v14 <= 0x40 )
          v15 = *((_QWORD *)v13 + 3) == 1;
        else
          v15 = v14 - 1 == (unsigned int)sub_C444A0((__int64)(v13 + 24));
        if ( v15 )
          return (unsigned __int8 *)a2;
      }
      else if ( *(_BYTE *)(v11 + 8) == 17 )
      {
        v31 = *(_DWORD *)(v11 + 32);
        if ( v31 )
        {
          v26 = 0;
          v27 = 0;
          while ( 1 )
          {
            v28 = sub_AD69F0((unsigned __int8 *)a3, v27);
            if ( !v28 )
              break;
            if ( *(_BYTE *)v28 != 13 )
            {
              if ( *(_BYTE *)v28 != 17 )
                break;
              v29 = *(_DWORD *)(v28 + 32);
              v26 = v29 <= 0x40 ? *(_QWORD *)(v28 + 24) == 1 : v29 - 1 == (unsigned int)sub_C444A0(v28 + 24);
              if ( !v26 )
                break;
            }
            if ( v31 == ++v27 )
            {
              if ( v26 )
                return (unsigned __int8 *)a2;
              break;
            }
          }
        }
      }
    }
  }
  v16 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17 <= 1
    && (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a3 + 8) + 8LL) - 17 > 1 )
  {
    v21 = *a1;
    v34 = 257;
    LODWORD(v32) = *(_DWORD *)(v16 + 32);
    BYTE4(v32) = *(_BYTE *)(v16 + 8) == 18;
    a3 = sub_B37620(v21, v32, a3, v33);
  }
  v17 = *a1;
  v34 = 257;
  v18 = (*(__int64 (__fastcall **)(unsigned int *, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v17[10] + 32LL))(
          v17[10],
          17,
          a2,
          a3,
          0,
          0);
  if ( !v18 )
    return sub_2AAE100((__int64 *)v17, 17, a2, a3, (__int64)v33, 0, 0);
  return (unsigned __int8 *)v18;
}
