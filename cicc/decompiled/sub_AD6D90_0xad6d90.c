// Function: sub_AD6D90
// Address: 0xad6d90
//
__int64 __fastcall sub_AD6D90(__int64 a1, _BYTE *a2)
{
  __int64 v3; // r13
  __int64 v5; // rsi
  char v6; // r15
  __int64 *v7; // rdi
  __int64 result; // rax
  unsigned __int64 v9; // r13
  __int64 *v10; // rax
  __int64 *v11; // rdx
  __int64 *v12; // rdi
  __int64 v13; // rsi
  __int64 *i; // rdx
  __int64 j; // r14
  _BYTE *v16; // rax
  _BYTE *v17; // r15
  unsigned int v18; // esi
  __int64 v19; // rsi
  __int64 **v20; // rdi
  char v21; // [rsp+Fh] [rbp-211h]
  __int64 v22; // [rsp+18h] [rbp-208h]
  _QWORD v23[2]; // [rsp+20h] [rbp-200h] BYREF
  __int64 **v24; // [rsp+30h] [rbp-1F0h] BYREF
  __int64 v25; // [rsp+38h] [rbp-1E8h]
  _BYTE v26[64]; // [rsp+40h] [rbp-1E0h] BYREF
  __int64 *v27; // [rsp+80h] [rbp-1A0h] BYREF
  __int64 v28; // [rsp+88h] [rbp-198h]
  __int64 v29; // [rsp+90h] [rbp-190h] BYREF
  int v30; // [rsp+98h] [rbp-188h]
  char v31; // [rsp+9Ch] [rbp-184h]
  char v32; // [rsp+A0h] [rbp-180h] BYREF
  __int64 *v33; // [rsp+E0h] [rbp-140h] BYREF
  __int64 v34; // [rsp+E8h] [rbp-138h]
  __int64 v35; // [rsp+F0h] [rbp-130h] BYREF
  int v36; // [rsp+F8h] [rbp-128h]
  char v37; // [rsp+FCh] [rbp-124h]
  char v38; // [rsp+100h] [rbp-120h] BYREF

  if ( (unsigned __int8)(*(_BYTE *)a1 - 12) <= 1u )
    return (__int64)a2;
  v3 = *(_QWORD *)(a1 + 8);
  if ( (unsigned __int8)(*(_BYTE *)a1 - 9) <= 2u )
  {
    v5 = a1;
    v33 = 0;
    v34 = (__int64)&v38;
    v27 = &v29;
    v35 = 8;
    v36 = 0;
    v37 = 1;
    v28 = 0x800000000LL;
    v24 = &v33;
    v25 = (__int64)&v27;
    v6 = sub_AA8FD0(&v24, a1);
    if ( v6 )
    {
      while ( 1 )
      {
        v7 = v27;
        if ( !(_DWORD)v28 )
          break;
        v5 = v27[(unsigned int)v28 - 1];
        LODWORD(v28) = v28 - 1;
        if ( !(unsigned __int8)sub_AA8FD0(&v24, v5) )
          goto LABEL_43;
      }
    }
    else
    {
LABEL_43:
      v7 = v27;
      v6 = 0;
    }
    if ( v7 != &v29 )
      _libc_free(v7, v5);
    if ( !v37 )
      _libc_free(v34, v5);
    if ( v6 )
      return (__int64)a2;
  }
  result = a1;
  if ( *(_BYTE *)(v3 + 8) == 17 )
  {
    v9 = *(unsigned int *)(v3 + 32);
    v10 = &v35;
    v11 = &v35;
    v12 = &v35;
    v33 = &v35;
    v13 = v9;
    v34 = 0x2000000000LL;
    if ( v9 )
    {
      if ( v9 > 0x20 )
      {
        sub_C8D5F0(&v33, &v35, v9, 8);
        v11 = v33;
        v10 = &v33[(unsigned int)v34];
      }
      for ( i = &v11[v9]; i != v10; ++v10 )
      {
        if ( v10 )
          *v10 = 0;
      }
      LODWORD(v34) = v9;
      for ( j = 0; j != v9; ++j )
      {
        v16 = (_BYTE *)sub_AD69F0((unsigned __int8 *)a1, (unsigned int)j);
        v17 = v16;
        if ( v16 )
        {
          v18 = (unsigned __int8)*v16;
          if ( (unsigned __int8)(*v16 - 12) <= 1u )
          {
            v17 = a2;
          }
          else if ( v18 > 8 && v18 <= 0xB )
          {
            v27 = 0;
            v28 = (__int64)&v32;
            v23[1] = &v24;
            v24 = (__int64 **)v26;
            v23[0] = &v27;
            v19 = (__int64)v16;
            v29 = 8;
            v30 = 0;
            v31 = 1;
            v25 = 0x800000000LL;
            v21 = sub_AA8FD0(v23, (__int64)v16);
            if ( v21 )
            {
              while ( 1 )
              {
                v20 = v24;
                if ( !(_DWORD)v25 )
                  break;
                v19 = (__int64)v24[(unsigned int)v25 - 1];
                LODWORD(v25) = v25 - 1;
                if ( !(unsigned __int8)sub_AA8FD0(v23, v19) )
                  goto LABEL_42;
              }
            }
            else
            {
LABEL_42:
              v21 = 0;
              v20 = v24;
            }
            if ( v20 != (__int64 **)v26 )
              _libc_free(v20, v19);
            if ( !v31 )
              _libc_free(v28, v19);
            if ( v21 )
              v17 = a2;
          }
        }
        v33[j] = (__int64)v17;
      }
      v12 = v33;
      v13 = (unsigned int)v34;
    }
    result = sub_AD3730(v12, v13);
    if ( v33 != &v35 )
    {
      v22 = result;
      _libc_free(v33, v13);
      return v22;
    }
  }
  return result;
}
