// Function: sub_1499CE0
// Address: 0x1499ce0
//
__int64 __fastcall sub_1499CE0(__int64 a1, __int64 a2)
{
  __int64 *v2; // r8
  __int64 v3; // rax
  _QWORD *v4; // rbx
  __int64 *v5; // rax
  _BYTE *v6; // rcx
  __int64 v7; // rdx
  _BYTE *v8; // rsi
  __int64 *v9; // r9
  __int64 *v10; // rdx
  __int64 *v11; // rdi
  __int64 *v12; // r15
  __int64 v13; // r13
  __int64 *v14; // r14
  unsigned int v15; // r12d
  __int64 *v17; // rax
  _QWORD *v18; // rax
  _BYTE *v19; // r12
  __int64 v20; // rbx
  _QWORD *v21; // r13
  unsigned __int64 v22; // rdi
  _BYTE *v23; // r11
  _QWORD *v24; // rax
  _QWORD *v25; // r10
  _QWORD *v26; // rax
  _QWORD *v27; // rsi
  unsigned __int64 v30; // [rsp+28h] [rbp-248h]
  __int64 v31; // [rsp+30h] [rbp-240h] BYREF
  __int64 *v32; // [rsp+38h] [rbp-238h]
  __int64 *v33; // [rsp+40h] [rbp-230h]
  __int64 v34; // [rsp+48h] [rbp-228h]
  int v35; // [rsp+50h] [rbp-220h]
  _BYTE v36[136]; // [rsp+58h] [rbp-218h] BYREF
  __int64 v37; // [rsp+E0h] [rbp-190h] BYREF
  _BYTE *v38; // [rsp+E8h] [rbp-188h]
  _BYTE *v39; // [rsp+F0h] [rbp-180h]
  __int64 v40; // [rsp+F8h] [rbp-178h]
  int v41; // [rsp+100h] [rbp-170h]
  _BYTE v42[136]; // [rsp+108h] [rbp-168h] BYREF
  __int64 v43; // [rsp+190h] [rbp-E0h] BYREF
  _BYTE *v44; // [rsp+198h] [rbp-D8h]
  _BYTE *v45; // [rsp+1A0h] [rbp-D0h]
  __int64 v46; // [rsp+1A8h] [rbp-C8h]
  int v47; // [rsp+1B0h] [rbp-C0h]
  _BYTE v48[184]; // [rsp+1B8h] [rbp-B8h] BYREF

  v2 = (__int64 *)v36;
  v3 = 8LL * *(unsigned int *)(a2 + 8);
  v32 = (__int64 *)v36;
  v4 = (_QWORD *)(a2 - v3);
  v31 = 0;
  v33 = (__int64 *)v36;
  v34 = 16;
  v35 = 0;
  if ( a2 == a2 - v3 )
    goto LABEL_23;
  v5 = (__int64 *)v36;
  do
  {
LABEL_5:
    v6 = (_BYTE *)*v4;
    if ( (unsigned __int8)(*(_BYTE *)*v4 - 4) <= 0x1Eu )
    {
      v7 = *((unsigned int *)v6 + 2);
      if ( (unsigned int)v7 > 1 )
      {
        v8 = *(_BYTE **)&v6[8 * (1 - v7)];
        if ( v8 )
        {
          if ( (unsigned __int8)(*v8 - 4) <= 0x1Eu )
          {
            if ( v2 != v5 )
              goto LABEL_3;
            v9 = &v2[HIDWORD(v34)];
            if ( v2 == v9 )
            {
LABEL_77:
              if ( HIDWORD(v34) >= (unsigned int)v34 )
              {
LABEL_3:
                sub_16CCBA0(&v31, v8);
                v5 = v33;
                v2 = v32;
                goto LABEL_4;
              }
              ++HIDWORD(v34);
              *v9 = (__int64)v8;
              v2 = v32;
              ++v31;
              v5 = v33;
            }
            else
            {
              v10 = v2;
              v11 = 0;
              while ( v8 != (_BYTE *)*v10 )
              {
                if ( *v10 == -2 )
                  v11 = v10;
                if ( v9 == ++v10 )
                {
                  if ( !v11 )
                    goto LABEL_77;
                  *v11 = (__int64)v8;
                  ++v4;
                  v5 = v33;
                  --v35;
                  v2 = v32;
                  ++v31;
                  if ( (_QWORD *)a2 != v4 )
                    goto LABEL_5;
                  goto LABEL_18;
                }
              }
            }
          }
        }
      }
    }
LABEL_4:
    ++v4;
  }
  while ( (_QWORD *)a2 != v4 );
LABEL_18:
  if ( v2 == v5 )
    v12 = &v5[HIDWORD(v34)];
  else
    v12 = &v5[(unsigned int)v34];
  while ( v12 != v5 )
  {
    v13 = *v5;
    v14 = v5;
    if ( (unsigned __int64)*v5 < 0xFFFFFFFFFFFFFFFELL )
    {
      if ( v12 != v5 )
      {
        while ( 1 )
        {
          v37 = 0;
          v40 = 16;
          v38 = v42;
          v39 = v42;
          v41 = 0;
          sub_1499BC0(a1, v13, (__int64)&v37);
          if ( HIDWORD(v40) != v41 )
          {
            v44 = v48;
            v45 = v48;
            v43 = 0;
            v46 = 16;
            v47 = 0;
            sub_1499BC0(a2, v13, (__int64)&v43);
            v18 = v39;
            if ( v39 == v38 )
              v19 = &v39[8 * HIDWORD(v40)];
            else
              v19 = &v39[8 * (unsigned int)v40];
            if ( v39 == v19 )
            {
LABEL_42:
              v22 = (unsigned __int64)v45;
LABEL_43:
              if ( (_BYTE *)v22 != v44 )
                _libc_free(v22);
              if ( v39 != v38 )
                _libc_free((unsigned __int64)v39);
              v15 = 0;
              goto LABEL_24;
            }
            while ( 1 )
            {
              v20 = *v18;
              v21 = v18;
              if ( *v18 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v19 == (_BYTE *)++v18 )
                goto LABEL_42;
            }
            v22 = (unsigned __int64)v45;
            if ( v19 == (_BYTE *)v18 )
              goto LABEL_43;
            v23 = v44;
            if ( v45 != v44 )
            {
LABEL_50:
              v30 = v22 + 8LL * (unsigned int)v46;
              v24 = (_QWORD *)sub_16CC9F0(&v43, v20);
              v25 = (_QWORD *)v30;
              if ( v20 == *v24 )
              {
                v22 = (unsigned __int64)v45;
                v23 = v44;
                if ( v45 == v44 )
                  v27 = &v45[8 * HIDWORD(v46)];
                else
                  v27 = &v45[8 * (unsigned int)v46];
              }
              else
              {
                v22 = (unsigned __int64)v45;
                v23 = v44;
                if ( v45 != v44 )
                {
                  v24 = &v45[8 * (unsigned int)v46];
                  goto LABEL_53;
                }
                v24 = &v45[8 * HIDWORD(v46)];
                v27 = v24;
              }
              goto LABEL_65;
            }
            while ( 1 )
            {
              v25 = (_QWORD *)(v22 + 8LL * HIDWORD(v46));
              if ( (_QWORD *)v22 == v25 )
              {
                v27 = (_QWORD *)v22;
                v24 = (_QWORD *)v22;
              }
              else
              {
                v24 = (_QWORD *)v22;
                do
                {
                  if ( v20 == *v24 )
                    break;
                  ++v24;
                }
                while ( v25 != v24 );
                v27 = (_QWORD *)(v22 + 8LL * HIDWORD(v46));
              }
LABEL_65:
              if ( v24 != v27 )
              {
                while ( *v24 >= 0xFFFFFFFFFFFFFFFELL )
                {
                  if ( v27 == ++v24 )
                  {
                    if ( v25 != v24 )
                      goto LABEL_54;
                    goto LABEL_69;
                  }
                }
              }
LABEL_53:
              if ( v25 == v24 )
                break;
LABEL_54:
              v26 = v21 + 1;
              if ( v21 + 1 == (_QWORD *)v19 )
                goto LABEL_43;
              while ( 1 )
              {
                v20 = *v26;
                v21 = v26;
                if ( *v26 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v19 == (_BYTE *)++v26 )
                  goto LABEL_43;
              }
              if ( v26 == (_QWORD *)v19 )
                goto LABEL_43;
              if ( (_BYTE *)v22 != v23 )
                goto LABEL_50;
            }
LABEL_69:
            if ( (_BYTE *)v22 != v23 )
              _libc_free(v22);
          }
          if ( v39 != v38 )
            _libc_free((unsigned __int64)v39);
          v17 = v14 + 1;
          if ( v14 + 1 == v12 )
            goto LABEL_23;
          v13 = *v17;
          ++v14;
          if ( (unsigned __int64)*v17 >= 0xFFFFFFFFFFFFFFFELL )
            break;
LABEL_35:
          if ( v12 == v14 )
            goto LABEL_23;
        }
        while ( v12 != ++v17 )
        {
          v13 = *v17;
          v14 = v17;
          if ( (unsigned __int64)*v17 < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_35;
        }
      }
      break;
    }
    ++v5;
  }
LABEL_23:
  v15 = 1;
LABEL_24:
  if ( v33 != v32 )
    _libc_free((unsigned __int64)v33);
  return v15;
}
