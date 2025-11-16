// Function: sub_1458EC0
// Address: 0x1458ec0
//
void __fastcall sub_1458EC0(__int64 a1, __int64 a2)
{
  char v2; // dl
  _BYTE *v3; // rsi
  __int64 v4; // rdi
  int v5; // eax
  _BYTE *v6; // r9
  _BYTE *v7; // rcx
  __int64 v8; // rbx
  _QWORD *v9; // r8
  _QWORD *v10; // rbx
  _QWORD *v11; // r14
  char v12; // dl
  _BYTE *v13; // rax
  __int64 v14; // rax
  __int64 v15; // r15
  _QWORD *v16; // rax
  _QWORD *v17; // rsi
  _QWORD *v18; // rcx
  __int64 v19; // rbx
  _QWORD *v20; // rax
  char v21; // dl
  __int64 v22; // rax
  unsigned __int64 v23; // rdi
  _QWORD *v24; // rax
  __int64 v25; // r14
  char v26; // dl
  _BYTE *v27; // rax
  __int64 v28; // rax
  char v29; // dl
  _QWORD *v30; // rsi
  _QWORD *v31; // rcx
  _QWORD *v32; // r8
  _QWORD *v33; // rcx
  _QWORD *v34; // r8
  _QWORD *v35; // rdx
  _QWORD *v36; // rsi
  __int64 v37; // [rsp+18h] [rbp-F8h] BYREF
  __int64 v38; // [rsp+20h] [rbp-F0h]
  _BYTE *v39; // [rsp+28h] [rbp-E8h] BYREF
  __int64 v40; // [rsp+30h] [rbp-E0h]
  _BYTE v41[64]; // [rsp+38h] [rbp-D8h] BYREF
  __int64 v42; // [rsp+78h] [rbp-98h] BYREF
  _BYTE *v43; // [rsp+80h] [rbp-90h]
  _BYTE *v44; // [rsp+88h] [rbp-88h]
  __int64 v45; // [rsp+90h] [rbp-80h]
  int v46; // [rsp+98h] [rbp-78h]
  _BYTE v47[112]; // [rsp+A0h] [rbp-70h] BYREF

  v39 = v41;
  v38 = a2;
  v40 = 0x800000000LL;
  v37 = a1;
  v42 = 0;
  v43 = v47;
  v44 = v47;
  v45 = 8;
  v46 = 0;
  sub_1412190((__int64)&v42, a1);
  if ( v2 )
  {
    if ( v37 == **(_QWORD **)(v38 + 8) )
      *(_BYTE *)v38 = 1;
    else
      sub_1458920((__int64)&v39, &v37);
  }
  v3 = v39;
  v4 = v38;
  v5 = v40;
  v6 = v39;
LABEL_5:
  v7 = &v3[8 * v5];
  if ( v5 )
  {
    while ( 2 )
    {
      if ( *(_BYTE *)v4 )
        goto LABEL_33;
      v8 = *((_QWORD *)v7 - 1);
      LODWORD(v40) = --v5;
      switch ( *(_WORD *)(v8 + 24) )
      {
        case 0:
        case 0xA:
          v7 -= 8;
          if ( v5 )
            continue;
          goto LABEL_33;
        case 1:
        case 2:
        case 3:
          v19 = *(_QWORD *)(v8 + 32);
          v20 = v43;
          if ( v44 != v43 )
            goto LABEL_27;
          v32 = &v43[8 * HIDWORD(v45)];
          if ( v43 == (_BYTE *)v32 )
            goto LABEL_82;
          v33 = 0;
          do
          {
            if ( v19 == *v20 )
              goto LABEL_50;
            if ( *v20 == -2 )
              v33 = v20;
            ++v20;
          }
          while ( v32 != v20 );
          if ( !v33 )
          {
LABEL_82:
            if ( HIDWORD(v45) >= (unsigned int)v45 )
            {
LABEL_27:
              sub_16CCBA0(&v42, v19);
              v4 = v38;
              if ( !v21 )
                goto LABEL_49;
            }
            else
            {
              ++HIDWORD(v45);
              *v32 = v19;
              v4 = v38;
              ++v42;
            }
            if ( v19 != **(_QWORD **)(v4 + 8) )
              goto LABEL_29;
          }
          else
          {
            *v33 = v19;
            v4 = v38;
            --v46;
            ++v42;
            if ( v19 != **(_QWORD **)(v38 + 8) )
              goto LABEL_29;
          }
          *(_BYTE *)v4 = 1;
          goto LABEL_48;
        case 4:
        case 5:
        case 7:
        case 8:
        case 9:
          v9 = *(_QWORD **)(v8 + 32);
          v10 = &v9[*(_QWORD *)(v8 + 40)];
          if ( v9 == v10 )
            goto LABEL_5;
          v11 = v9;
          break;
        case 6:
          v23 = (unsigned __int64)v44;
          v24 = v43;
          v25 = *(_QWORD *)(v8 + 32);
          if ( v44 != v43 )
            goto LABEL_39;
          v34 = &v44[8 * HIDWORD(v45)];
          if ( v44 == (_BYTE *)v34 )
            goto LABEL_80;
          v35 = v44;
          v36 = 0;
          do
          {
            if ( v25 == *v35 )
              goto LABEL_44;
            if ( *v35 == -2 )
              v36 = v35;
            ++v35;
          }
          while ( v34 != v35 );
          if ( v36 )
          {
            *v36 = v25;
            v27 = (_BYTE *)v38;
            --v46;
            ++v42;
            if ( v25 == **(_QWORD **)(v38 + 8) )
              goto LABEL_78;
          }
          else
          {
LABEL_80:
            if ( HIDWORD(v45) >= (unsigned int)v45 )
            {
LABEL_39:
              sub_16CCBA0(&v42, *(_QWORD *)(v8 + 32));
              v23 = (unsigned __int64)v44;
              v24 = v43;
              if ( !v26 )
                goto LABEL_44;
            }
            else
            {
              ++HIDWORD(v45);
              *v34 = v25;
              ++v42;
            }
            v27 = (_BYTE *)v38;
            if ( v25 == **(_QWORD **)(v38 + 8) )
            {
LABEL_78:
              *v27 = 1;
              v23 = (unsigned __int64)v44;
              v24 = v43;
              goto LABEL_44;
            }
          }
          v28 = (unsigned int)v40;
          if ( (unsigned int)v40 >= HIDWORD(v40) )
          {
            sub_16CD150(&v39, v41, 0, 8);
            v28 = (unsigned int)v40;
          }
          *(_QWORD *)&v39[8 * v28] = v25;
          v23 = (unsigned __int64)v44;
          LODWORD(v40) = v40 + 1;
          v24 = v43;
LABEL_44:
          v19 = *(_QWORD *)(v8 + 40);
          if ( v24 != (_QWORD *)v23 )
            goto LABEL_45;
          v30 = &v24[HIDWORD(v45)];
          if ( v24 == v30 )
          {
LABEL_84:
            if ( HIDWORD(v45) >= (unsigned int)v45 )
            {
LABEL_45:
              sub_16CCBA0(&v42, v19);
              if ( !v29 )
                goto LABEL_48;
            }
            else
            {
              ++HIDWORD(v45);
              *v30 = v19;
              ++v42;
            }
LABEL_46:
            if ( v19 != **(_QWORD **)(v38 + 8) )
            {
LABEL_29:
              v22 = (unsigned int)v40;
              if ( (unsigned int)v40 >= HIDWORD(v40) )
              {
                sub_16CD150(&v39, v41, 0, 8);
                v22 = (unsigned int)v40;
              }
              *(_QWORD *)&v39[8 * v22] = v19;
              v3 = v39;
              v4 = v38;
              v5 = v40 + 1;
              LODWORD(v40) = v40 + 1;
              v6 = v39;
              goto LABEL_5;
            }
            *(_BYTE *)v38 = 1;
          }
          else
          {
            v31 = 0;
            while ( v19 != *v24 )
            {
              if ( *v24 == -2 )
                v31 = v24;
              if ( v30 == ++v24 )
              {
                if ( !v31 )
                  goto LABEL_84;
                *v31 = v19;
                --v46;
                ++v42;
                goto LABEL_46;
              }
            }
          }
LABEL_48:
          v4 = v38;
LABEL_49:
          v3 = v39;
LABEL_50:
          v5 = v40;
          v6 = v3;
          goto LABEL_5;
      }
      break;
    }
    while ( 1 )
    {
      v15 = *v11;
      v16 = v43;
      if ( v44 == v43 )
      {
        v17 = &v43[8 * HIDWORD(v45)];
        if ( v43 != (_BYTE *)v17 )
        {
          v18 = 0;
          while ( v15 != *v16 )
          {
            if ( *v16 == -2 )
              v18 = v16;
            if ( v17 == ++v16 )
            {
              if ( !v18 )
                goto LABEL_51;
              *v18 = v15;
              v13 = (_BYTE *)v38;
              --v46;
              ++v42;
              if ( v15 != **(_QWORD **)(v38 + 8) )
                goto LABEL_12;
              goto LABEL_25;
            }
          }
          goto LABEL_15;
        }
LABEL_51:
        if ( HIDWORD(v45) < (unsigned int)v45 )
        {
          ++HIDWORD(v45);
          *v17 = v15;
          ++v42;
LABEL_11:
          v13 = (_BYTE *)v38;
          if ( v15 == **(_QWORD **)(v38 + 8) )
          {
LABEL_25:
            *v13 = 1;
          }
          else
          {
LABEL_12:
            v14 = (unsigned int)v40;
            if ( (unsigned int)v40 >= HIDWORD(v40) )
            {
              sub_16CD150(&v39, v41, 0, 8);
              v14 = (unsigned int)v40;
            }
            *(_QWORD *)&v39[8 * v14] = v15;
            LODWORD(v40) = v40 + 1;
          }
          goto LABEL_15;
        }
      }
      sub_16CCBA0(&v42, *v11);
      if ( v12 )
        goto LABEL_11;
LABEL_15:
      if ( v10 == ++v11 )
        goto LABEL_48;
    }
  }
LABEL_33:
  if ( v44 != v43 )
  {
    _libc_free((unsigned __int64)v44);
    v6 = v39;
  }
  if ( v6 != v41 )
    _libc_free((unsigned __int64)v6);
}
