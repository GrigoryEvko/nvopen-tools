// Function: sub_145B270
// Address: 0x145b270
//
void __fastcall sub_145B270(__int64 a1, __int64 a2, __int64 a3)
{
  char v3; // dl
  _BYTE *v4; // rsi
  int v5; // eax
  _BYTE *v6; // rdi
  _BYTE *v7; // rcx
  __int64 v8; // rbx
  _QWORD *v9; // r8
  _QWORD *v10; // rbx
  _QWORD *v11; // r15
  char v12; // dl
  __int64 v13; // r14
  __int64 *v14; // rax
  __int64 *v15; // rsi
  __int64 *v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // r10
  _QWORD *v19; // rax
  __int64 v20; // rax
  __int64 v21; // r8
  __int64 *v22; // rax
  char v23; // dl
  __int64 v24; // r8
  unsigned __int64 v25; // rdi
  __int64 *v26; // rax
  char v27; // dl
  __int64 v28; // rsi
  char v29; // dl
  _QWORD *v30; // rdi
  unsigned int v31; // r11d
  _QWORD *v32; // rcx
  __int64 *v33; // rdi
  __int64 *v34; // rcx
  __int64 *v35; // rdi
  __int64 *v36; // rcx
  __int64 *v37; // r9
  __int64 *v38; // rdx
  __int64 *v39; // rsi
  __int64 v40; // [rsp+10h] [rbp-100h] BYREF
  __int64 v41; // [rsp+18h] [rbp-F8h] BYREF
  __int64 *v42; // [rsp+20h] [rbp-F0h]
  _BYTE *v43; // [rsp+28h] [rbp-E8h] BYREF
  __int64 v44; // [rsp+30h] [rbp-E0h]
  _BYTE v45[64]; // [rsp+38h] [rbp-D8h] BYREF
  __int64 v46; // [rsp+78h] [rbp-98h] BYREF
  __int64 *v47; // [rsp+80h] [rbp-90h]
  __int64 *v48; // [rsp+88h] [rbp-88h]
  __int64 v49; // [rsp+90h] [rbp-80h]
  int v50; // [rsp+98h] [rbp-78h]
  _BYTE v51[112]; // [rsp+A0h] [rbp-70h] BYREF

  v42 = &v40;
  v43 = v45;
  v44 = 0x800000000LL;
  v40 = a3;
  v46 = 0;
  v47 = (__int64 *)v51;
  v48 = (__int64 *)v51;
  v49 = 8;
  v50 = 0;
  v41 = a2;
  sub_1412190((__int64)&v46, a2);
  if ( v3 )
  {
    if ( *(_WORD *)(v41 + 24) == 7 )
      sub_1412190(*v42, *(_QWORD *)(v41 + 48));
    sub_1458920((__int64)&v43, &v41);
  }
  v4 = v43;
  v5 = v44;
  v6 = v43;
LABEL_3:
  v7 = &v4[8 * v5];
  if ( v5 )
  {
    while ( 2 )
    {
      v8 = *((_QWORD *)v7 - 1);
      LODWORD(v44) = --v5;
      switch ( *(_WORD *)(v8 + 24) )
      {
        case 0:
        case 0xA:
          v7 -= 8;
          if ( v5 )
            continue;
          goto LABEL_27;
        case 1:
        case 2:
        case 3:
          v21 = *(_QWORD *)(v8 + 32);
          v22 = v47;
          v41 = v21;
          if ( v48 == v47 )
          {
            v33 = &v47[HIDWORD(v49)];
            if ( v47 != v33 )
            {
              v34 = 0;
              do
              {
                if ( v21 == *v22 )
                  goto LABEL_25;
                if ( *v22 == -2 )
                  v34 = v22;
                ++v22;
              }
              while ( v33 != v22 );
              if ( v34 )
              {
                *v34 = v21;
                --v50;
                ++v46;
                goto LABEL_54;
              }
            }
            if ( HIDWORD(v49) < (unsigned int)v49 )
            {
              ++HIDWORD(v49);
              *v33 = v21;
              ++v46;
              goto LABEL_54;
            }
          }
          sub_16CCBA0(&v46, v21);
          v4 = v43;
          if ( v23 )
            goto LABEL_54;
          goto LABEL_25;
        case 4:
        case 5:
        case 7:
        case 8:
        case 9:
          v9 = *(_QWORD **)(v8 + 32);
          v10 = &v9[*(_QWORD *)(v8 + 40)];
          if ( v9 == v10 )
            goto LABEL_3;
          v11 = v9;
          break;
        case 6:
          v24 = *(_QWORD *)(v8 + 32);
          v25 = (unsigned __int64)v48;
          v26 = v47;
          v41 = v24;
          if ( v48 != v47 )
            goto LABEL_33;
          v37 = &v48[HIDWORD(v49)];
          if ( v48 == v37 )
            goto LABEL_82;
          v38 = v48;
          v39 = 0;
          while ( v24 != *v38 )
          {
            if ( *v38 == -2 )
              v39 = v38;
            if ( v37 == ++v38 )
            {
              if ( v39 )
              {
                *v39 = v24;
                --v50;
                ++v46;
              }
              else
              {
LABEL_82:
                if ( HIDWORD(v49) >= (unsigned int)v49 )
                {
LABEL_33:
                  sub_16CCBA0(&v46, v24);
                  v25 = (unsigned __int64)v48;
                  v26 = v47;
                  if ( !v27 )
                    break;
                }
                else
                {
                  ++HIDWORD(v49);
                  *v37 = v24;
                  ++v46;
                }
              }
              if ( *(_WORD *)(v41 + 24) == 7 )
                sub_1412190(*v42, *(_QWORD *)(v41 + 48));
              sub_1458920((__int64)&v43, &v41);
              v25 = (unsigned __int64)v48;
              v26 = v47;
              break;
            }
          }
          v28 = *(_QWORD *)(v8 + 40);
          v41 = v28;
          if ( (__int64 *)v25 != v26 )
            goto LABEL_35;
          v35 = &v26[HIDWORD(v49)];
          if ( v35 == v26 )
          {
LABEL_84:
            if ( HIDWORD(v49) >= (unsigned int)v49 )
            {
LABEL_35:
              sub_16CCBA0(&v46, v28);
              if ( !v29 )
                goto LABEL_36;
            }
            else
            {
              ++HIDWORD(v49);
              *v35 = v28;
              ++v46;
            }
LABEL_54:
            if ( *(_WORD *)(v41 + 24) == 7 )
              sub_1412190(*v42, *(_QWORD *)(v41 + 48));
            sub_1458920((__int64)&v43, &v41);
          }
          else
          {
            v36 = 0;
            while ( v28 != *v26 )
            {
              if ( *v26 == -2 )
                v36 = v26;
              if ( v35 == ++v26 )
              {
                if ( !v36 )
                  goto LABEL_84;
                *v36 = v28;
                --v50;
                ++v46;
                goto LABEL_54;
              }
            }
          }
LABEL_36:
          v4 = v43;
LABEL_25:
          v5 = v44;
          v6 = v4;
          goto LABEL_3;
      }
      break;
    }
    while ( 1 )
    {
      v13 = *v11;
      v14 = v47;
      if ( v48 == v47 )
      {
        v15 = &v47[HIDWORD(v49)];
        if ( v47 != v15 )
        {
          v16 = 0;
          while ( v13 != *v14 )
          {
            if ( *v14 == -2 )
              v16 = v14;
            if ( v15 == ++v14 )
            {
              if ( !v16 )
                goto LABEL_57;
              *v16 = v13;
              --v50;
              ++v46;
              goto LABEL_18;
            }
          }
          goto LABEL_8;
        }
LABEL_57:
        if ( HIDWORD(v49) < (unsigned int)v49 )
          break;
      }
      sub_16CCBA0(&v46, *v11);
      if ( v12 )
        goto LABEL_18;
LABEL_8:
      if ( v10 == ++v11 )
        goto LABEL_36;
    }
    ++HIDWORD(v49);
    *v15 = v13;
    ++v46;
LABEL_18:
    if ( *(_WORD *)(v13 + 24) == 7 )
    {
      v17 = *(_QWORD *)(v13 + 48);
      v18 = *v42;
      v19 = *(_QWORD **)(*v42 + 8);
      if ( *(_QWORD **)(*v42 + 16) != v19 )
        goto LABEL_20;
      v30 = &v19[*(unsigned int *)(v18 + 28)];
      v31 = *(_DWORD *)(v18 + 28);
      if ( v19 != v30 )
      {
        v32 = 0;
        while ( v17 != *v19 )
        {
          if ( *v19 == -2 )
            v32 = v19;
          if ( v30 == ++v19 )
          {
            if ( !v32 )
              goto LABEL_78;
            *v32 = v17;
            ++*(_QWORD *)v18;
            v20 = (unsigned int)v44;
            --*(_DWORD *)(v18 + 32);
            if ( (unsigned int)v20 < HIDWORD(v44) )
              goto LABEL_22;
            goto LABEL_45;
          }
        }
        goto LABEL_21;
      }
LABEL_78:
      if ( v31 < *(_DWORD *)(v18 + 24) )
      {
        *(_DWORD *)(v18 + 28) = v31 + 1;
        *v30 = v17;
        ++*(_QWORD *)v18;
      }
      else
      {
LABEL_20:
        sub_16CCBA0(*v42, v17);
      }
    }
LABEL_21:
    v20 = (unsigned int)v44;
    if ( (unsigned int)v44 >= HIDWORD(v44) )
    {
LABEL_45:
      sub_16CD150(&v43, v45, 0, 8);
      v20 = (unsigned int)v44;
    }
LABEL_22:
    *(_QWORD *)&v43[8 * v20] = v13;
    LODWORD(v44) = v44 + 1;
    goto LABEL_8;
  }
LABEL_27:
  if ( v48 != v47 )
  {
    _libc_free((unsigned __int64)v48);
    v6 = v43;
  }
  if ( v6 != v45 )
    _libc_free((unsigned __int64)v6);
}
