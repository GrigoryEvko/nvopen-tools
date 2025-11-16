// Function: sub_AD7180
// Address: 0xad7180
//
__int64 __fastcall sub_AD7180(_BYTE *a1, unsigned __int8 *a2)
{
  __int64 v2; // r15
  __int64 v4; // rsi
  char v5; // r12
  __int64 *v6; // rdi
  __int64 v7; // r12
  __int64 **v8; // rax
  unsigned __int64 v9; // rcx
  __int64 *v10; // rax
  int v11; // ebx
  __int64 *v12; // rdx
  __int64 *i; // rdx
  __int64 v14; // r13
  __int64 v15; // rax
  unsigned __int8 *v16; // r12
  __int64 v17; // rsi
  int v18; // ecx
  __int64 **v19; // rdi
  char v21; // bl
  __int64 v22; // rax
  __int64 **v23; // [rsp+8h] [rbp-228h]
  char v24; // [rsp+1Eh] [rbp-212h]
  char v25; // [rsp+1Fh] [rbp-211h]
  unsigned __int64 v26; // [rsp+28h] [rbp-208h]
  _QWORD v27[2]; // [rsp+30h] [rbp-200h] BYREF
  __int64 **v28; // [rsp+40h] [rbp-1F0h] BYREF
  __int64 v29; // [rsp+48h] [rbp-1E8h]
  _BYTE v30[64]; // [rsp+50h] [rbp-1E0h] BYREF
  __int64 *v31; // [rsp+90h] [rbp-1A0h] BYREF
  __int64 v32; // [rsp+98h] [rbp-198h]
  __int64 v33; // [rsp+A0h] [rbp-190h] BYREF
  int v34; // [rsp+A8h] [rbp-188h]
  char v35; // [rsp+ACh] [rbp-184h]
  char v36; // [rsp+B0h] [rbp-180h] BYREF
  __int64 *v37; // [rsp+F0h] [rbp-140h] BYREF
  __int64 v38; // [rsp+F8h] [rbp-138h]
  __int64 v39; // [rsp+100h] [rbp-130h] BYREF
  int v40; // [rsp+108h] [rbp-128h]
  char v41; // [rsp+10Ch] [rbp-124h]
  char v42; // [rsp+110h] [rbp-120h] BYREF

  v2 = (__int64)a1;
  if ( (unsigned __int8)(*a1 - 12) <= 1u )
    return v2;
  if ( (unsigned __int8)(*a1 - 9) <= 2u )
  {
    v4 = (__int64)a1;
    v37 = 0;
    v32 = 0x800000000LL;
    v38 = (__int64)&v42;
    v39 = 8;
    v40 = 0;
    v41 = 1;
    v31 = &v33;
    v28 = &v37;
    v29 = (__int64)&v31;
    v5 = sub_AA8FD0(&v28, (__int64)a1);
    if ( v5 )
    {
      while ( 1 )
      {
        v6 = v31;
        if ( !(_DWORD)v32 )
          break;
        v4 = v31[(unsigned int)v32 - 1];
        LODWORD(v32) = v32 - 1;
        if ( !(unsigned __int8)sub_AA8FD0(&v28, v4) )
          goto LABEL_44;
      }
    }
    else
    {
LABEL_44:
      v6 = v31;
      v5 = 0;
    }
    if ( v6 != &v33 )
      _libc_free(v6, v4);
    if ( !v41 )
      _libc_free(v38, v4);
    if ( v5 )
      return v2;
  }
  v7 = *(_QWORD *)(v2 + 8);
  if ( !(unsigned __int8)sub_AC2BE0(a2) )
  {
    if ( *(_BYTE *)(v7 + 8) == 17 )
    {
      v8 = *(__int64 ***)(v7 + 24);
      v9 = *(unsigned int *)(v7 + 32);
      v38 = 0x2000000000LL;
      v23 = v8;
      v10 = &v39;
      v11 = v9;
      v26 = v9;
      v37 = &v39;
      if ( v9 )
      {
        v12 = &v39;
        if ( v9 > 0x20 )
        {
          sub_C8D5F0(&v37, &v39, v9, 8);
          v12 = v37;
          v10 = &v37[(unsigned int)v38];
        }
        for ( i = &v12[v26]; i != v10; ++v10 )
        {
          if ( v10 )
            *v10 = 0;
        }
        v25 = 0;
        LODWORD(v38) = v11;
        v14 = 0;
        do
        {
          v15 = sub_AD69F0((unsigned __int8 *)v2, (unsigned int)v14);
          v37[v14] = v15;
          v16 = (unsigned __int8 *)sub_AD69F0(a2, (unsigned int)v14);
          v17 = v37[v14];
          v18 = *(unsigned __int8 *)v17;
          if ( (_BYTE)v18 != 12 && v18 != 13 )
          {
            if ( (unsigned __int8)(*(_BYTE *)v17 - 9) > 2u )
              goto LABEL_40;
            v31 = 0;
            v28 = (__int64 **)v30;
            v29 = 0x800000000LL;
            v32 = (__int64)&v36;
            v33 = 8;
            v34 = 0;
            v35 = 1;
            v27[0] = &v31;
            v27[1] = &v28;
            v24 = sub_AA8FD0(v27, v17);
            if ( v24 )
            {
              while ( 1 )
              {
                v19 = v28;
                if ( !(_DWORD)v29 )
                  break;
                v17 = (__int64)v28[(unsigned int)v29 - 1];
                LODWORD(v29) = v29 - 1;
                if ( !(unsigned __int8)sub_AA8FD0(v27, v17) )
                  goto LABEL_43;
              }
            }
            else
            {
LABEL_43:
              v24 = 0;
              v19 = v28;
            }
            if ( v19 != (__int64 **)v30 )
              _libc_free(v19, v17);
            if ( !v35 )
              _libc_free(v32, v17);
            if ( !v24 )
            {
LABEL_40:
              v21 = sub_AC2BE0(v16);
              if ( v21 )
              {
                v22 = sub_ACA8A0(v23);
                v25 = v21;
                v37[v14] = v22;
              }
            }
          }
          ++v14;
        }
        while ( v14 != v26 );
        if ( v25 )
        {
          v17 = (unsigned int)v38;
          v2 = sub_AD3730(v37, (unsigned int)v38);
        }
        if ( v37 != &v39 )
          _libc_free(v37, v17);
      }
    }
    return v2;
  }
  return sub_ACA8A0((__int64 **)v7);
}
