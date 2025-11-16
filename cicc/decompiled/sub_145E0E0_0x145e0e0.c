// Function: sub_145E0E0
// Address: 0x145e0e0
//
void __fastcall sub_145E0E0(__int64 a1, __int64 a2)
{
  char v2; // dl
  _BYTE *v3; // rbx
  __int64 v4; // r14
  _BYTE *v5; // rsi
  int v6; // eax
  _BYTE *v7; // rdi
  _BYTE *v8; // rcx
  __int64 v9; // rbx
  __int64 *v10; // r8
  __int64 *v11; // rbx
  __int64 *v12; // r15
  char v13; // dl
  _BYTE *v14; // r14
  __int64 v15; // r12
  __int64 *v16; // rax
  __int64 *v17; // rsi
  __int64 *v18; // rcx
  __int64 v19; // r8
  __int64 *v20; // rax
  char v21; // dl
  __int64 v22; // r8
  unsigned __int64 v23; // rdi
  __int64 *v24; // rax
  char v25; // dl
  _BYTE *v26; // r14
  __int64 v27; // rsi
  char v28; // dl
  __int64 v29; // rax
  __int64 *v30; // rdi
  __int64 *v31; // rcx
  __int64 *v32; // r9
  __int64 *v33; // rdx
  __int64 *v34; // rsi
  __int64 *v35; // rdi
  __int64 *v36; // rcx
  __int64 v37; // [rsp+10h] [rbp-100h] BYREF
  __int64 v38; // [rsp+18h] [rbp-F8h] BYREF
  __int64 v39; // [rsp+20h] [rbp-F0h]
  _BYTE *v40; // [rsp+28h] [rbp-E8h] BYREF
  __int64 v41; // [rsp+30h] [rbp-E0h]
  _BYTE v42[64]; // [rsp+38h] [rbp-D8h] BYREF
  __int64 v43; // [rsp+78h] [rbp-98h] BYREF
  __int64 *v44; // [rsp+80h] [rbp-90h]
  __int64 *v45; // [rsp+88h] [rbp-88h]
  __int64 v46; // [rsp+90h] [rbp-80h]
  int v47; // [rsp+98h] [rbp-78h]
  _BYTE v48[112]; // [rsp+A0h] [rbp-70h] BYREF

  v40 = v42;
  v39 = a2;
  v41 = 0x800000000LL;
  v37 = a1;
  v43 = 0;
  v44 = (__int64 *)v48;
  v45 = (__int64 *)v48;
  v46 = 8;
  v47 = 0;
  sub_1412190((__int64)&v43, a1);
  if ( v2 )
  {
LABEL_2:
    v3 = (_BYTE *)v39;
    v38 = v37;
    if ( (*(unsigned __int8 (__fastcall **)(__int64 *))(v39 + 8))(&v38) )
      *v3 = 1;
    else
      sub_1458920((__int64)&v40, &v37);
  }
LABEL_4:
  v4 = v39;
LABEL_5:
  v5 = v40;
LABEL_6:
  v6 = v41;
  v7 = v5;
LABEL_7:
  v8 = &v5[8 * v6];
  if ( v6 )
  {
    while ( 2 )
    {
      if ( !*(_BYTE *)v4 )
      {
        v9 = *((_QWORD *)v8 - 1);
        LODWORD(v41) = --v6;
        switch ( *(_WORD *)(v9 + 24) )
        {
          case 0:
          case 0xA:
            v8 -= 8;
            if ( v6 )
              continue;
            goto LABEL_31;
          case 1:
          case 2:
          case 3:
            v19 = *(_QWORD *)(v9 + 32);
            v20 = v44;
            v37 = v19;
            if ( v45 != v44 )
              goto LABEL_26;
            v30 = &v44[HIDWORD(v46)];
            if ( v44 == v30 )
              goto LABEL_76;
            v31 = 0;
            do
            {
              if ( v19 == *v20 )
                goto LABEL_6;
              if ( *v20 == -2 )
                v31 = v20;
              ++v20;
            }
            while ( v30 != v20 );
            if ( !v31 )
            {
LABEL_76:
              if ( HIDWORD(v46) >= (unsigned int)v46 )
              {
LABEL_26:
                sub_16CCBA0(&v43, v19);
                v4 = v39;
                if ( !v21 )
                  goto LABEL_5;
              }
              else
              {
                ++HIDWORD(v46);
                *v30 = v19;
                v4 = v39;
                ++v43;
              }
            }
            else
            {
              *v31 = v19;
              v4 = v39;
              --v47;
              ++v43;
            }
            v38 = v37;
            if ( (*(unsigned __int8 (__fastcall **)(__int64 *))(v4 + 8))(&v38) )
              *(_BYTE *)v4 = 1;
            else
              sub_1458920((__int64)&v40, &v37);
            v4 = v39;
            v5 = v40;
            goto LABEL_6;
          case 4:
          case 5:
          case 7:
          case 8:
          case 9:
            v10 = *(__int64 **)(v9 + 32);
            v11 = &v10[*(_QWORD *)(v9 + 40)];
            if ( v10 == v11 )
              goto LABEL_7;
            v12 = v10;
            break;
          case 6:
            v22 = *(_QWORD *)(v9 + 32);
            v23 = (unsigned __int64)v45;
            v24 = v44;
            v37 = v22;
            if ( v45 != v44 )
              goto LABEL_37;
            v32 = &v45[HIDWORD(v46)];
            if ( v45 == v32 )
              goto LABEL_80;
            v33 = v45;
            v34 = 0;
            while ( v22 != *v33 )
            {
              if ( *v33 == -2 )
                v34 = v33;
              if ( v32 == ++v33 )
              {
                if ( v34 )
                {
                  *v34 = v22;
                  --v47;
                  ++v43;
                }
                else
                {
LABEL_80:
                  if ( HIDWORD(v46) >= (unsigned int)v46 )
                  {
LABEL_37:
                    sub_16CCBA0(&v43, v22);
                    v23 = (unsigned __int64)v45;
                    v24 = v44;
                    if ( !v25 )
                      break;
                  }
                  else
                  {
                    ++HIDWORD(v46);
                    *v32 = v22;
                    ++v43;
                  }
                }
                v26 = (_BYTE *)v39;
                v38 = v37;
                if ( (*(unsigned __int8 (__fastcall **)(__int64 *))(v39 + 8))(&v38) )
                  *v26 = 1;
                else
                  sub_1458920((__int64)&v40, &v37);
                v23 = (unsigned __int64)v45;
                v24 = v44;
                break;
              }
            }
            v27 = *(_QWORD *)(v9 + 40);
            v37 = v27;
            if ( (__int64 *)v23 != v24 )
              goto LABEL_42;
            v35 = &v24[HIDWORD(v46)];
            if ( v35 != v24 )
            {
              v36 = 0;
              while ( v27 != *v24 )
              {
                if ( *v24 == -2 )
                  v36 = v24;
                if ( v35 == ++v24 )
                {
                  if ( !v36 )
                    goto LABEL_78;
                  *v36 = v27;
                  --v47;
                  ++v43;
                  goto LABEL_2;
                }
              }
              goto LABEL_4;
            }
LABEL_78:
            if ( HIDWORD(v46) >= (unsigned int)v46 )
            {
LABEL_42:
              sub_16CCBA0(&v43, v27);
              if ( !v28 )
                goto LABEL_4;
            }
            else
            {
              ++HIDWORD(v46);
              *v35 = v27;
              ++v43;
            }
            goto LABEL_2;
        }
        while ( 1 )
        {
          v15 = *v12;
          v16 = v44;
          if ( v45 == v44 )
          {
            v17 = &v44[HIDWORD(v46)];
            if ( v44 != v17 )
            {
              v18 = 0;
              while ( v15 != *v16 )
              {
                if ( *v16 == -2 )
                  v18 = v16;
                if ( v17 == ++v16 )
                {
                  if ( !v18 )
                    goto LABEL_47;
                  *v18 = v15;
                  --v47;
                  ++v43;
                  goto LABEL_13;
                }
              }
              goto LABEL_15;
            }
LABEL_47:
            if ( HIDWORD(v46) < (unsigned int)v46 )
              break;
          }
          sub_16CCBA0(&v43, *v12);
          if ( v13 )
            goto LABEL_13;
LABEL_15:
          if ( v11 == ++v12 )
            goto LABEL_4;
        }
        ++HIDWORD(v46);
        *v17 = v15;
        ++v43;
LABEL_13:
        v14 = (_BYTE *)v39;
        v38 = v15;
        if ( (*(unsigned __int8 (__fastcall **)(__int64 *))(v39 + 8))(&v38) )
        {
          *v14 = 1;
        }
        else
        {
          v29 = (unsigned int)v41;
          if ( (unsigned int)v41 >= HIDWORD(v41) )
          {
            sub_16CD150(&v40, v42, 0, 8);
            v29 = (unsigned int)v41;
          }
          *(_QWORD *)&v40[8 * v29] = v15;
          LODWORD(v41) = v41 + 1;
        }
        goto LABEL_15;
      }
      break;
    }
  }
LABEL_31:
  if ( v45 != v44 )
  {
    _libc_free((unsigned __int64)v45);
    v7 = v40;
  }
  if ( v7 != v42 )
    _libc_free((unsigned __int64)v7);
}
