// Function: sub_145A080
// Address: 0x145a080
//
void __fastcall sub_145A080(__int64 a1, _BYTE *a2)
{
  char v2; // dl
  _BYTE *v3; // rsi
  _BYTE *v4; // rdi
  int v5; // eax
  _BYTE *v6; // r9
  _BYTE *v7; // rcx
  __int64 v8; // rbx
  _QWORD *v9; // r8
  _QWORD *v10; // rbx
  _QWORD *v11; // r14
  char v12; // dl
  __int64 v13; // rax
  __int64 v14; // r15
  _QWORD *v15; // rax
  _QWORD *v16; // rsi
  _QWORD *v17; // rcx
  __int64 v18; // rbx
  _QWORD *v19; // rax
  char v20; // dl
  __int64 v21; // rax
  unsigned __int64 v22; // rdi
  _QWORD *v23; // rax
  __int64 v24; // r14
  char v25; // dl
  __int64 v26; // rax
  char v27; // dl
  _QWORD *v28; // r8
  _QWORD *v29; // rdx
  _QWORD *v30; // rsi
  _QWORD *v31; // r8
  _QWORD *v32; // rcx
  _QWORD *v33; // rsi
  _QWORD *v34; // rcx
  __int64 v35; // [rsp+18h] [rbp-F8h] BYREF
  _BYTE *v36; // [rsp+20h] [rbp-F0h]
  _BYTE *v37; // [rsp+28h] [rbp-E8h] BYREF
  __int64 v38; // [rsp+30h] [rbp-E0h]
  _BYTE v39[64]; // [rsp+38h] [rbp-D8h] BYREF
  __int64 v40; // [rsp+78h] [rbp-98h] BYREF
  _BYTE *v41; // [rsp+80h] [rbp-90h]
  _BYTE *v42; // [rsp+88h] [rbp-88h]
  __int64 v43; // [rsp+90h] [rbp-80h]
  int v44; // [rsp+98h] [rbp-78h]
  _BYTE v45[112]; // [rsp+A0h] [rbp-70h] BYREF

  v37 = v39;
  v36 = a2;
  v38 = 0x800000000LL;
  v35 = a1;
  v40 = 0;
  v41 = v45;
  v42 = v45;
  v43 = 8;
  v44 = 0;
  sub_1412190((__int64)&v40, a1);
  if ( v2 )
  {
    if ( *(_WORD *)(v35 + 24) == 10 && !*(_QWORD *)(v35 - 8) )
      *v36 = 1;
    else
      sub_1458920((__int64)&v37, &v35);
  }
  v3 = v37;
  v4 = v36;
  v5 = v38;
  v6 = v37;
LABEL_6:
  v7 = &v3[8 * v5];
  if ( v5 )
  {
    while ( 2 )
    {
      if ( !*v4 )
      {
        v8 = *((_QWORD *)v7 - 1);
        LODWORD(v38) = --v5;
        switch ( *(_WORD *)(v8 + 24) )
        {
          case 0:
          case 0xA:
            v7 -= 8;
            if ( v5 )
              continue;
            goto LABEL_35;
          case 1:
          case 2:
          case 3:
            v18 = *(_QWORD *)(v8 + 32);
            v19 = v41;
            if ( v42 != v41 )
              goto LABEL_28;
            v31 = &v41[8 * HIDWORD(v43)];
            if ( v41 == (_BYTE *)v31 )
              goto LABEL_87;
            v32 = 0;
            do
            {
              if ( v18 == *v19 )
                goto LABEL_54;
              if ( *v19 == -2 )
                v32 = v19;
              ++v19;
            }
            while ( v31 != v19 );
            if ( !v32 )
            {
LABEL_87:
              if ( HIDWORD(v43) >= (unsigned int)v43 )
              {
LABEL_28:
                sub_16CCBA0(&v40, v18);
                v4 = v36;
                if ( !v20 )
                  goto LABEL_53;
              }
              else
              {
                ++HIDWORD(v43);
                *v31 = v18;
                v4 = v36;
                ++v40;
              }
            }
            else
            {
              *v32 = v18;
              v4 = v36;
              --v44;
              ++v40;
            }
            if ( *(_WORD *)(v18 + 24) == 10 && !*(_QWORD *)(v18 - 8) )
            {
              *v4 = 1;
              v4 = v36;
              goto LABEL_53;
            }
            goto LABEL_31;
          case 4:
          case 5:
          case 7:
          case 8:
          case 9:
            v9 = *(_QWORD **)(v8 + 32);
            v10 = &v9[*(_QWORD *)(v8 + 40)];
            if ( v9 == v10 )
              goto LABEL_6;
            v11 = v9;
            break;
          case 6:
            v22 = (unsigned __int64)v42;
            v23 = v41;
            v24 = *(_QWORD *)(v8 + 32);
            if ( v42 != v41 )
              goto LABEL_41;
            v28 = &v42[8 * HIDWORD(v43)];
            if ( v42 == (_BYTE *)v28 )
              goto LABEL_89;
            v29 = v42;
            v30 = 0;
            while ( v24 != *v29 )
            {
              if ( *v29 == -2 )
                v30 = v29;
              if ( v28 == ++v29 )
              {
                if ( v30 )
                {
                  *v30 = v24;
                  --v44;
                  ++v40;
                }
                else
                {
LABEL_89:
                  if ( HIDWORD(v43) >= (unsigned int)v43 )
                  {
LABEL_41:
                    sub_16CCBA0(&v40, *(_QWORD *)(v8 + 32));
                    v22 = (unsigned __int64)v42;
                    v23 = v41;
                    if ( !v25 )
                      break;
                  }
                  else
                  {
                    ++HIDWORD(v43);
                    *v28 = v24;
                    ++v40;
                  }
                }
                if ( *(_WORD *)(v24 + 24) == 10 && !*(_QWORD *)(v24 - 8) )
                {
                  *v36 = 1;
                  v22 = (unsigned __int64)v42;
                  v23 = v41;
                }
                else
                {
                  v26 = (unsigned int)v38;
                  if ( (unsigned int)v38 >= HIDWORD(v38) )
                  {
                    sub_16CD150(&v37, v39, 0, 8);
                    v26 = (unsigned int)v38;
                  }
                  *(_QWORD *)&v37[8 * v26] = v24;
                  v22 = (unsigned __int64)v42;
                  LODWORD(v38) = v38 + 1;
                  v23 = v41;
                }
                break;
              }
            }
            v18 = *(_QWORD *)(v8 + 40);
            if ( v23 != (_QWORD *)v22 )
              goto LABEL_48;
            v33 = &v23[HIDWORD(v43)];
            if ( v23 == v33 )
            {
LABEL_85:
              if ( HIDWORD(v43) >= (unsigned int)v43 )
              {
LABEL_48:
                sub_16CCBA0(&v40, v18);
                if ( !v27 )
                  goto LABEL_52;
              }
              else
              {
                ++HIDWORD(v43);
                *v33 = v18;
                ++v40;
              }
LABEL_49:
              if ( *(_WORD *)(v18 + 24) != 10 || *(_QWORD *)(v18 - 8) )
              {
LABEL_31:
                v21 = (unsigned int)v38;
                if ( (unsigned int)v38 >= HIDWORD(v38) )
                {
                  sub_16CD150(&v37, v39, 0, 8);
                  v21 = (unsigned int)v38;
                }
                *(_QWORD *)&v37[8 * v21] = v18;
                v3 = v37;
                v4 = v36;
                v5 = v38 + 1;
                LODWORD(v38) = v38 + 1;
                v6 = v37;
                goto LABEL_6;
              }
              *v36 = 1;
            }
            else
            {
              v34 = 0;
              while ( v18 != *v23 )
              {
                if ( *v23 == -2 )
                  v34 = v23;
                if ( v33 == ++v23 )
                {
                  if ( !v34 )
                    goto LABEL_85;
                  *v34 = v18;
                  --v44;
                  ++v40;
                  goto LABEL_49;
                }
              }
            }
LABEL_52:
            v4 = v36;
LABEL_53:
            v3 = v37;
LABEL_54:
            v5 = v38;
            v6 = v3;
            goto LABEL_6;
        }
        while ( 1 )
        {
          v14 = *v11;
          v15 = v41;
          if ( v42 == v41 )
          {
            v16 = &v41[8 * HIDWORD(v43)];
            if ( v41 != (_BYTE *)v16 )
            {
              v17 = 0;
              while ( v14 != *v15 )
              {
                if ( *v15 == -2 )
                  v17 = v15;
                if ( v16 == ++v15 )
                {
                  if ( !v17 )
                    goto LABEL_56;
                  *v17 = v14;
                  --v44;
                  ++v40;
                  goto LABEL_12;
                }
              }
              goto LABEL_17;
            }
LABEL_56:
            if ( HIDWORD(v43) < (unsigned int)v43 )
              break;
          }
          sub_16CCBA0(&v40, *v11);
          if ( v12 )
            goto LABEL_12;
LABEL_17:
          if ( v10 == ++v11 )
            goto LABEL_52;
        }
        ++HIDWORD(v43);
        *v16 = v14;
        ++v40;
LABEL_12:
        if ( *(_WORD *)(v14 + 24) == 10 && !*(_QWORD *)(v14 - 8) )
        {
          *v36 = 1;
        }
        else
        {
          v13 = (unsigned int)v38;
          if ( (unsigned int)v38 >= HIDWORD(v38) )
          {
            sub_16CD150(&v37, v39, 0, 8);
            v13 = (unsigned int)v38;
          }
          *(_QWORD *)&v37[8 * v13] = v14;
          LODWORD(v38) = v38 + 1;
        }
        goto LABEL_17;
      }
      break;
    }
  }
LABEL_35:
  if ( v42 != v41 )
  {
    _libc_free((unsigned __int64)v42);
    v6 = v37;
  }
  if ( v6 != v39 )
    _libc_free((unsigned __int64)v6);
}
