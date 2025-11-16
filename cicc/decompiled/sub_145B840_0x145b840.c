// Function: sub_145B840
// Address: 0x145b840
//
__int64 __fastcall sub_145B840(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // rcx
  char v5; // dl
  _BYTE *v6; // rsi
  int v7; // eax
  _BYTE *v8; // r9
  _BYTE *v9; // rdi
  __int64 v10; // rbx
  _QWORD *v11; // r8
  _QWORD *v12; // rbx
  _QWORD *v13; // r15
  char v14; // dl
  __int64 v15; // r14
  _QWORD *v16; // rax
  _QWORD *v17; // rsi
  _QWORD *v18; // rcx
  _QWORD *v19; // rax
  _QWORD *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // r8
  _QWORD *v23; // rax
  char v24; // dl
  unsigned int v25; // r12d
  __int64 v27; // rsi
  unsigned __int8 v28; // al
  __int64 v29; // r8
  unsigned __int64 v30; // rdi
  _QWORD *v31; // rax
  char v32; // dl
  __int64 v33; // rsi
  char v34; // dl
  _QWORD *v35; // r9
  _QWORD *v36; // rdi
  _QWORD *v37; // rax
  _QWORD *v38; // rdx
  __int64 v39; // rsi
  unsigned __int8 v40; // al
  _QWORD *v41; // rdi
  _QWORD *v42; // rcx
  _QWORD *v43; // r9
  _QWORD *v44; // rdx
  _QWORD *v45; // rsi
  _WORD *v46; // rcx
  _QWORD *v47; // rax
  _QWORD *v48; // rdx
  __int64 v49; // rsi
  unsigned __int8 v50; // dl
  _QWORD *v51; // rax
  _QWORD *v52; // rdx
  __int64 v53; // rsi
  unsigned __int8 v54; // al
  _QWORD *v55; // rax
  _QWORD *v56; // rdx
  char v57; // al
  char v58; // al
  char v59; // al
  __int64 v60; // rsi
  unsigned __int8 v61; // al
  char v62; // al
  _WORD *v63; // [rsp+0h] [rbp-140h]
  _QWORD *v64; // [rsp+0h] [rbp-140h]
  _WORD *v65; // [rsp+0h] [rbp-140h]
  _QWORD *v66; // [rsp+0h] [rbp-140h]
  _QWORD *v67; // [rsp+0h] [rbp-140h]
  __int64 v68; // [rsp+10h] [rbp-130h] BYREF
  __int64 v69; // [rsp+18h] [rbp-128h] BYREF
  __int64 v70; // [rsp+20h] [rbp-120h] BYREF
  __int64 v71; // [rsp+28h] [rbp-118h] BYREF
  _QWORD v72[4]; // [rsp+30h] [rbp-110h] BYREF
  _QWORD *v73; // [rsp+50h] [rbp-F0h]
  _BYTE *v74; // [rsp+58h] [rbp-E8h] BYREF
  __int64 v75; // [rsp+60h] [rbp-E0h]
  _BYTE v76[64]; // [rsp+68h] [rbp-D8h] BYREF
  __int64 v77; // [rsp+A8h] [rbp-98h] BYREF
  _BYTE *v78; // [rsp+B0h] [rbp-90h]
  _BYTE *v79; // [rsp+B8h] [rbp-88h]
  __int64 v80; // [rsp+C0h] [rbp-80h]
  int v81; // [rsp+C8h] [rbp-78h]
  _BYTE v82[112]; // [rsp+D0h] [rbp-70h] BYREF

  v73 = v72;
  v74 = v76;
  v72[1] = a1;
  v75 = 0x800000000LL;
  v72[2] = a4;
  v72[0] = 256;
  v72[3] = a2;
  v77 = 0;
  v78 = v82;
  v79 = v82;
  v80 = 8;
  v81 = 0;
  v68 = a3;
  sub_1412190((__int64)&v77, a3);
  v4 = v73;
  if ( v5 )
  {
    switch ( *(_WORD *)(v68 + 24) )
    {
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
      case 8:
      case 9:
        goto LABEL_118;
      case 6:
      case 0xB:
        goto LABEL_128;
      case 7:
        v55 = (_QWORD *)v73[1];
        if ( !v55 )
          goto LABEL_128;
        v56 = *(_QWORD **)(v68 + 48);
        if ( v56 == v55 )
          goto LABEL_118;
        break;
      case 0xA:
        v60 = *(_QWORD *)(v68 - 8);
        v61 = *(_BYTE *)(v60 + 16);
        if ( v61 == 17 )
          goto LABEL_2;
        if ( v61 > 0x17u )
        {
          v67 = v73;
          v62 = sub_15CCE20(v73[3], v60, v73[2]);
          v4 = v67;
          if ( v62 )
            goto LABEL_119;
        }
        goto LABEL_128;
    }
    while ( 1 )
    {
      v55 = (_QWORD *)*v55;
      if ( v56 == v55 )
        break;
      if ( !v55 )
      {
LABEL_128:
        *(_WORD *)v4 = 1;
        v4 = v73;
        goto LABEL_2;
      }
    }
LABEL_118:
    sub_1458920((__int64)&v74, &v68);
LABEL_119:
    v4 = v73;
  }
LABEL_2:
  v6 = v74;
  v7 = v75;
  v8 = v74;
  while ( 1 )
  {
    v9 = &v6[8 * v7];
    if ( !v7 )
      break;
LABEL_4:
    if ( *(_BYTE *)v4 )
      break;
    v10 = *((_QWORD *)v9 - 1);
    LODWORD(v75) = --v7;
    switch ( *(_WORD *)(v10 + 24) )
    {
      case 0:
      case 0xA:
        v9 -= 8;
        if ( !v7 )
          goto LABEL_33;
        goto LABEL_4;
      case 1:
      case 2:
      case 3:
        v22 = *(_QWORD *)(v10 + 32);
        v23 = v78;
        v69 = v22;
        if ( v79 != v78 )
          goto LABEL_30;
        v35 = &v78[8 * HIDWORD(v80)];
        if ( v78 == (_BYTE *)v35 )
          goto LABEL_106;
        v36 = 0;
        while ( v22 != *v23 )
        {
          if ( *v23 == -2 )
            v36 = v23;
          if ( v35 == ++v23 )
          {
            if ( v36 )
            {
              *v36 = v22;
              v4 = v73;
              --v81;
              ++v77;
            }
            else
            {
LABEL_106:
              if ( HIDWORD(v80) >= (unsigned int)v80 )
              {
LABEL_30:
                sub_16CCBA0(&v77, v22);
                v4 = v73;
                v6 = v74;
                if ( !v24 )
                  goto LABEL_31;
              }
              else
              {
                ++HIDWORD(v80);
                *v35 = v22;
                v4 = v73;
                ++v77;
              }
            }
            switch ( *(_WORD *)(v69 + 24) )
            {
              case 0:
              case 1:
              case 2:
              case 3:
              case 4:
              case 5:
              case 8:
              case 9:
                goto LABEL_64;
              case 6:
              case 0xB:
                goto LABEL_67;
              case 7:
                v37 = (_QWORD *)v4[1];
                if ( !v37 )
                  goto LABEL_67;
                v38 = *(_QWORD **)(v69 + 48);
                if ( v38 == v37 )
                  goto LABEL_64;
                break;
              case 0xA:
                v39 = *(_QWORD *)(v69 - 8);
                v40 = *(_BYTE *)(v39 + 16);
                if ( v40 == 17 )
                  goto LABEL_45;
                if ( v40 > 0x17u )
                {
                  v64 = v4;
                  v57 = sub_15CCE20(v4[3], v39, v4[2]);
                  v4 = v64;
                  if ( v57 )
                    goto LABEL_28;
                }
                goto LABEL_67;
              default:
                goto LABEL_131;
            }
            do
            {
              v37 = (_QWORD *)*v37;
              if ( v38 == v37 )
              {
LABEL_64:
                sub_1458920((__int64)&v74, &v69);
                v4 = v73;
                goto LABEL_45;
              }
            }
            while ( v37 );
LABEL_67:
            *(_WORD *)v4 = 1;
            v4 = v73;
            v6 = v74;
            goto LABEL_31;
          }
        }
        goto LABEL_31;
      case 4:
      case 5:
      case 7:
      case 8:
      case 9:
        v11 = *(_QWORD **)(v10 + 32);
        v12 = &v11[*(_QWORD *)(v10 + 40)];
        if ( v11 == v12 )
          continue;
        v13 = v11;
        do
        {
          while ( 2 )
          {
            v15 = *v13;
            v16 = v78;
            if ( v79 != v78 )
              goto LABEL_8;
            v17 = &v78[8 * HIDWORD(v80)];
            if ( v78 != (_BYTE *)v17 )
            {
              v18 = 0;
              while ( v15 != *v16 )
              {
                if ( *v16 == -2 )
                  v18 = v16;
                if ( v17 == ++v16 )
                {
                  if ( !v18 )
                    goto LABEL_48;
                  *v18 = v15;
                  --v81;
                  ++v77;
                  goto LABEL_19;
                }
              }
LABEL_9:
              if ( v12 == ++v13 )
                goto LABEL_28;
              continue;
            }
            break;
          }
LABEL_48:
          if ( HIDWORD(v80) < (unsigned int)v80 )
          {
            ++HIDWORD(v80);
            *v17 = v15;
            ++v77;
          }
          else
          {
LABEL_8:
            sub_16CCBA0(&v77, *v13);
            if ( !v14 )
              goto LABEL_9;
          }
LABEL_19:
          switch ( *(_WORD *)(v15 + 24) )
          {
            case 0:
            case 1:
            case 2:
            case 3:
            case 4:
            case 5:
            case 8:
            case 9:
              goto LABEL_25;
            case 6:
            case 0xB:
              goto LABEL_40;
            case 7:
              v19 = (_QWORD *)v73[1];
              if ( !v19 )
                goto LABEL_40;
              v20 = *(_QWORD **)(v15 + 48);
              if ( v20 == v19 )
                goto LABEL_25;
              break;
            case 0xA:
              v27 = *(_QWORD *)(v15 - 8);
              v28 = *(_BYTE *)(v27 + 16);
              if ( v28 == 17 )
                goto LABEL_9;
              if ( v28 <= 0x17u )
                goto LABEL_40;
              v63 = v73;
              if ( !(unsigned __int8)sub_15CCE20(v73[3], v27, v73[2]) )
                *v63 = 1;
              goto LABEL_9;
            default:
              goto LABEL_131;
          }
          while ( 1 )
          {
            v19 = (_QWORD *)*v19;
            if ( v20 == v19 )
              break;
            if ( !v19 )
            {
LABEL_40:
              *(_WORD *)v73 = 1;
              goto LABEL_9;
            }
          }
LABEL_25:
          v21 = (unsigned int)v75;
          if ( (unsigned int)v75 >= HIDWORD(v75) )
          {
            sub_16CD150(&v74, v76, 0, 8);
            v21 = (unsigned int)v75;
          }
          ++v13;
          *(_QWORD *)&v74[8 * v21] = v15;
          LODWORD(v75) = v75 + 1;
        }
        while ( v12 != v13 );
        goto LABEL_28;
      case 6:
        v29 = *(_QWORD *)(v10 + 32);
        v30 = (unsigned __int64)v79;
        v31 = v78;
        v71 = v29;
        if ( v79 != v78 )
          goto LABEL_42;
        v43 = &v79[8 * HIDWORD(v80)];
        if ( v79 == (_BYTE *)v43 )
          goto LABEL_108;
        v44 = v79;
        v45 = 0;
        break;
      default:
LABEL_131:
        BUG();
    }
    while ( v29 != *v44 )
    {
      if ( *v44 == -2 )
        v45 = v44;
      if ( v43 == ++v44 )
      {
        if ( v45 )
        {
          *v45 = v29;
          v30 = (unsigned __int64)v79;
          --v81;
          v31 = v78;
          ++v77;
        }
        else
        {
LABEL_108:
          if ( HIDWORD(v80) >= (unsigned int)v80 )
          {
LABEL_42:
            sub_16CCBA0(&v77, v29);
            v30 = (unsigned __int64)v79;
            v31 = v78;
            if ( !v32 )
              break;
          }
          else
          {
            ++HIDWORD(v80);
            *v43 = v29;
            v31 = v78;
            ++v77;
            v30 = (unsigned __int64)v79;
          }
        }
        v46 = v73;
        switch ( *(_WORD *)(v71 + 24) )
        {
          case 0:
          case 1:
          case 2:
          case 3:
          case 4:
          case 5:
          case 8:
          case 9:
            goto LABEL_89;
          case 6:
          case 0xB:
            goto LABEL_93;
          case 7:
            v47 = (_QWORD *)v73[1];
            if ( !v47 )
              goto LABEL_93;
            v48 = *(_QWORD **)(v71 + 48);
            if ( v48 == v47 )
              goto LABEL_89;
            break;
          case 0xA:
            v49 = *(_QWORD *)(v71 - 8);
            v50 = *(_BYTE *)(v49 + 16);
            if ( v50 == 17 )
              goto LABEL_43;
            if ( v50 > 0x17u )
            {
              v65 = v73;
              v58 = sub_15CCE20(v73[3], v49, v73[2]);
              v46 = v65;
              if ( v58 )
                goto LABEL_90;
            }
            goto LABEL_93;
          default:
            goto LABEL_131;
        }
        while ( 1 )
        {
          v47 = (_QWORD *)*v47;
          if ( v48 == v47 )
            break;
          if ( !v47 )
          {
LABEL_93:
            *v46 = 1;
            v30 = (unsigned __int64)v79;
            v31 = v78;
            goto LABEL_43;
          }
        }
LABEL_89:
        sub_1458920((__int64)&v74, &v71);
LABEL_90:
        v30 = (unsigned __int64)v79;
        v31 = v78;
        break;
      }
    }
LABEL_43:
    v33 = *(_QWORD *)(v10 + 40);
    v70 = v33;
    if ( (_QWORD *)v30 != v31 )
    {
LABEL_44:
      sub_16CCBA0(&v77, v33);
      v4 = v73;
      if ( !v34 )
        goto LABEL_45;
      goto LABEL_96;
    }
    v41 = &v31[HIDWORD(v80)];
    if ( v41 == v31 )
      goto LABEL_110;
    v42 = 0;
    do
    {
      if ( v33 == *v31 )
      {
LABEL_28:
        v4 = v73;
        goto LABEL_45;
      }
      if ( *v31 == -2 )
        v42 = v31;
      ++v31;
    }
    while ( v41 != v31 );
    if ( !v42 )
    {
LABEL_110:
      if ( HIDWORD(v80) >= (unsigned int)v80 )
        goto LABEL_44;
      ++HIDWORD(v80);
      *v41 = v33;
      v4 = v73;
      ++v77;
    }
    else
    {
      *v42 = v33;
      v4 = v73;
      --v81;
      ++v77;
    }
LABEL_96:
    switch ( *(_WORD *)(v70 + 24) )
    {
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
      case 8:
      case 9:
        goto LABEL_102;
      case 6:
      case 0xB:
        goto LABEL_105;
      case 7:
        v51 = (_QWORD *)v4[1];
        if ( !v51 )
          goto LABEL_105;
        v52 = *(_QWORD **)(v70 + 48);
        if ( v52 == v51 )
          goto LABEL_102;
        break;
      case 0xA:
        v53 = *(_QWORD *)(v70 - 8);
        v54 = *(_BYTE *)(v53 + 16);
        if ( v54 == 17 )
          goto LABEL_45;
        if ( v54 > 0x17u )
        {
          v66 = v4;
          v59 = sub_15CCE20(v4[3], v53, v4[2]);
          v4 = v66;
          if ( v59 )
            goto LABEL_28;
        }
        goto LABEL_105;
      default:
        goto LABEL_131;
    }
    do
    {
      v51 = (_QWORD *)*v51;
      if ( v52 == v51 )
      {
LABEL_102:
        sub_1458920((__int64)&v74, &v70);
        v4 = v73;
        goto LABEL_45;
      }
    }
    while ( v51 );
LABEL_105:
    *(_WORD *)v4 = 1;
    v4 = v73;
LABEL_45:
    v6 = v74;
LABEL_31:
    v7 = v75;
    v8 = v6;
  }
LABEL_33:
  v25 = BYTE1(v72[0]);
  if ( v79 != v78 )
  {
    _libc_free((unsigned __int64)v79);
    v8 = v74;
  }
  if ( v8 != v76 )
    _libc_free((unsigned __int64)v8);
  return v25;
}
