// Function: sub_145AC20
// Address: 0x145ac20
//
void __fastcall sub_145AC20(__int64 a1, _BYTE *a2)
{
  char v2; // dl
  __int16 v3; // dx
  bool v4; // al
  _BYTE *v5; // rsi
  _BYTE *v6; // rdi
  int v7; // eax
  _BYTE *v8; // r9
  _BYTE *v9; // rcx
  __int64 v10; // rbx
  _QWORD *v11; // r8
  _QWORD *v12; // rbx
  _QWORD *v13; // r13
  char v14; // dl
  __int16 v15; // ax
  __int64 v16; // r15
  __int64 *v17; // rax
  __int64 *v18; // rsi
  __int64 *v19; // rcx
  __int64 v20; // rax
  __int64 v21; // r8
  __int64 *v22; // rax
  char v23; // dl
  __int16 v24; // dx
  bool v25; // al
  __int64 v26; // r8
  unsigned __int64 v27; // rdi
  __int64 *v28; // rax
  char v29; // dl
  __int16 v30; // dx
  __int64 v31; // rsi
  char v32; // dl
  __int16 v33; // dx
  __int64 *v34; // rdi
  __int64 *v35; // rcx
  __int64 *v36; // r9
  __int64 *v37; // rdx
  __int64 *v38; // rsi
  __int64 *v39; // r9
  __int64 *v40; // rcx
  __int64 v41; // [rsp+18h] [rbp-F8h] BYREF
  _BYTE *v42; // [rsp+20h] [rbp-F0h]
  _BYTE *v43; // [rsp+28h] [rbp-E8h] BYREF
  __int64 v44; // [rsp+30h] [rbp-E0h]
  _BYTE v45[64]; // [rsp+38h] [rbp-D8h] BYREF
  __int64 v46; // [rsp+78h] [rbp-98h] BYREF
  __int64 *v47; // [rsp+80h] [rbp-90h]
  __int64 *v48; // [rsp+88h] [rbp-88h]
  __int64 v49; // [rsp+90h] [rbp-80h]
  int v50; // [rsp+98h] [rbp-78h]
  _BYTE v51[112]; // [rsp+A0h] [rbp-70h] BYREF

  v43 = v45;
  v42 = a2;
  v44 = 0x800000000LL;
  v41 = a1;
  v46 = 0;
  v47 = (__int64 *)v51;
  v48 = (__int64 *)v51;
  v49 = 8;
  v50 = 0;
  sub_1412190((__int64)&v46, a1);
  if ( !v2 )
    goto LABEL_6;
  v3 = *(_WORD *)(v41 + 24);
  if ( v3 == 10 )
  {
    v4 = *(_BYTE *)(*(_QWORD *)(v41 - 8) + 16LL) == 9;
  }
  else
  {
    if ( v3 )
      goto LABEL_99;
    v4 = *(_BYTE *)(*(_QWORD *)(v41 + 32) + 16LL) == 9;
  }
  if ( !v4 )
  {
LABEL_99:
    sub_1458920((__int64)&v43, &v41);
    goto LABEL_6;
  }
  *v42 = 1;
LABEL_6:
  v5 = v43;
  v6 = v42;
  v7 = v44;
  v8 = v43;
LABEL_7:
  v9 = &v5[8 * v7];
  if ( v7 )
  {
    while ( 2 )
    {
      if ( *v6 )
        goto LABEL_41;
      v10 = *((_QWORD *)v9 - 1);
      LODWORD(v44) = --v7;
      switch ( *(_WORD *)(v10 + 24) )
      {
        case 0:
        case 0xA:
          v9 -= 8;
          if ( v7 )
            continue;
          goto LABEL_41;
        case 1:
        case 2:
        case 3:
          v21 = *(_QWORD *)(v10 + 32);
          v22 = v47;
          v41 = v21;
          if ( v48 != v47 )
            goto LABEL_32;
          v39 = &v47[HIDWORD(v49)];
          if ( v47 == v39 )
            goto LABEL_93;
          v40 = 0;
          do
          {
            if ( v21 == *v22 )
              goto LABEL_39;
            if ( *v22 == -2 )
              v40 = v22;
            ++v22;
          }
          while ( v39 != v22 );
          if ( !v40 )
          {
LABEL_93:
            if ( HIDWORD(v49) >= (unsigned int)v49 )
            {
LABEL_32:
              sub_16CCBA0(&v46, v21);
              v6 = v42;
              if ( !v23 )
                goto LABEL_38;
            }
            else
            {
              ++HIDWORD(v49);
              *v39 = v21;
              v6 = v42;
              ++v46;
            }
          }
          else
          {
            *v40 = v21;
            v6 = v42;
            --v50;
            ++v46;
          }
          v24 = *(_WORD *)(v41 + 24);
          if ( v24 == 10 )
          {
            v25 = *(_BYTE *)(*(_QWORD *)(v41 - 8) + 16LL) == 9;
          }
          else
          {
            if ( v24 )
            {
LABEL_90:
              sub_1458920((__int64)&v43, &v41);
              v6 = v42;
              v5 = v43;
              goto LABEL_39;
            }
            v25 = *(_BYTE *)(*(_QWORD *)(v41 + 32) + 16LL) == 9;
          }
          if ( v25 )
          {
            *v6 = 1;
            goto LABEL_37;
          }
          goto LABEL_90;
        case 4:
        case 5:
        case 7:
        case 8:
        case 9:
          v11 = *(_QWORD **)(v10 + 32);
          v12 = &v11[*(_QWORD *)(v10 + 40)];
          if ( v11 == v12 )
            goto LABEL_7;
          v13 = v11;
          break;
        case 6:
          v26 = *(_QWORD *)(v10 + 32);
          v27 = (unsigned __int64)v48;
          v28 = v47;
          v41 = v26;
          if ( v48 != v47 )
            goto LABEL_47;
          v36 = &v48[HIDWORD(v49)];
          if ( v48 == v36 )
            goto LABEL_97;
          v37 = v48;
          v38 = 0;
          do
          {
            if ( v26 == *v37 )
              goto LABEL_51;
            if ( *v37 == -2 )
              v38 = v37;
            ++v37;
          }
          while ( v36 != v37 );
          if ( v38 )
          {
            *v38 = v26;
            --v50;
            ++v46;
          }
          else
          {
LABEL_97:
            if ( HIDWORD(v49) >= (unsigned int)v49 )
            {
LABEL_47:
              sub_16CCBA0(&v46, v26);
              v27 = (unsigned __int64)v48;
              v28 = v47;
              if ( !v29 )
                goto LABEL_51;
            }
            else
            {
              ++HIDWORD(v49);
              *v36 = v26;
              ++v46;
            }
          }
          v30 = *(_WORD *)(v41 + 24);
          if ( v30 == 10 )
          {
            if ( *(_BYTE *)(*(_QWORD *)(v41 - 8) + 16LL) == 9 )
              goto LABEL_50;
          }
          else if ( !v30 && *(_BYTE *)(*(_QWORD *)(v41 + 32) + 16LL) == 9 )
          {
LABEL_50:
            *v42 = 1;
            v27 = (unsigned __int64)v48;
            v28 = v47;
            goto LABEL_51;
          }
          sub_1458920((__int64)&v43, &v41);
          v27 = (unsigned __int64)v48;
          v28 = v47;
LABEL_51:
          v31 = *(_QWORD *)(v10 + 40);
          v41 = v31;
          if ( v28 != (__int64 *)v27 )
            goto LABEL_52;
          v34 = &v28[HIDWORD(v49)];
          if ( v34 == v28 )
          {
LABEL_95:
            if ( HIDWORD(v49) >= (unsigned int)v49 )
            {
LABEL_52:
              sub_16CCBA0(&v46, v31);
              if ( !v32 )
                goto LABEL_37;
            }
            else
            {
              ++HIDWORD(v49);
              *v34 = v31;
              ++v46;
            }
LABEL_53:
            v33 = *(_WORD *)(v41 + 24);
            if ( v33 == 10 )
            {
              if ( *(_BYTE *)(*(_QWORD *)(v41 - 8) + 16LL) == 9 )
                goto LABEL_55;
            }
            else if ( !v33 && *(_BYTE *)(*(_QWORD *)(v41 + 32) + 16LL) == 9 )
            {
LABEL_55:
              *v42 = 1;
              goto LABEL_37;
            }
            sub_1458920((__int64)&v43, &v41);
          }
          else
          {
            v35 = 0;
            while ( v31 != *v28 )
            {
              if ( *v28 == -2 )
                v35 = v28;
              if ( v34 == ++v28 )
              {
                if ( !v35 )
                  goto LABEL_95;
                *v35 = v31;
                --v50;
                ++v46;
                goto LABEL_53;
              }
            }
          }
LABEL_37:
          v6 = v42;
LABEL_38:
          v5 = v43;
LABEL_39:
          v7 = v44;
          v8 = v5;
          goto LABEL_7;
      }
      break;
    }
    while ( 1 )
    {
      v16 = *v13;
      v17 = v47;
      if ( v48 != v47 )
        goto LABEL_12;
      v18 = &v47[HIDWORD(v49)];
      if ( v47 != v18 )
      {
        v19 = 0;
        while ( v16 != *v17 )
        {
          if ( *v17 == -2 )
            v19 = v17;
          if ( v18 == ++v17 )
          {
            if ( !v19 )
              goto LABEL_56;
            *v19 = v16;
            --v50;
            ++v46;
            v15 = *(_WORD *)(v16 + 24);
            if ( v15 == 10 )
              goto LABEL_14;
            goto LABEL_26;
          }
        }
        goto LABEL_16;
      }
LABEL_56:
      if ( HIDWORD(v49) < (unsigned int)v49 )
      {
        ++HIDWORD(v49);
        *v18 = v16;
        ++v46;
      }
      else
      {
LABEL_12:
        sub_16CCBA0(&v46, *v13);
        if ( !v14 )
          goto LABEL_16;
      }
      v15 = *(_WORD *)(v16 + 24);
      if ( v15 == 10 )
      {
LABEL_14:
        if ( *(_BYTE *)(*(_QWORD *)(v16 - 8) + 16LL) != 9 )
          goto LABEL_28;
      }
      else
      {
LABEL_26:
        if ( v15 || *(_BYTE *)(*(_QWORD *)(v16 + 32) + 16LL) != 9 )
        {
LABEL_28:
          v20 = (unsigned int)v44;
          if ( (unsigned int)v44 >= HIDWORD(v44) )
          {
            sub_16CD150(&v43, v45, 0, 8);
            v20 = (unsigned int)v44;
          }
          *(_QWORD *)&v43[8 * v20] = v16;
          LODWORD(v44) = v44 + 1;
          goto LABEL_16;
        }
      }
      *v42 = 1;
LABEL_16:
      if ( v12 == ++v13 )
        goto LABEL_37;
    }
  }
LABEL_41:
  if ( v48 != v47 )
  {
    _libc_free((unsigned __int64)v48);
    v8 = v43;
  }
  if ( v8 != v45 )
    _libc_free((unsigned __int64)v8);
}
