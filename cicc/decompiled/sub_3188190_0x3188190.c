// Function: sub_3188190
// Address: 0x3188190
//
__int64 __fastcall sub_3188190(__int64 a1, __int64 a2)
{
  __int64 v4; // rbx
  char v5; // al
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 *v13; // rbx
  __int64 *v14; // r15
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rdi
  __int64 v58; // r15
  __int64 v59; // rbx
  __int64 v60; // rsi
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rdi
  __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // rax
  int v74; // r15d
  int v75; // r15d
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 v78; // [rsp+0h] [rbp-70h] BYREF
  __int64 v79; // [rsp+8h] [rbp-68h] BYREF
  _BYTE v80[16]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v81; // [rsp+20h] [rbp-50h]
  char v82; // [rsp+30h] [rbp-40h]

  v78 = a2;
  v79 = 0;
  sub_3187ED0((__int64)v80, a1 + 88, &v78, &v79);
  if ( v79 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v79 + 8LL))(v79);
  v4 = v81;
  if ( !v82 )
    return *(_QWORD *)(v4 + 8);
  v5 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 <= 0x1Cu )
  {
    switch ( v5 )
    {
      case 0:
        v49 = sub_22077B0(0x20u);
        v7 = v49;
        if ( v49 )
        {
          sub_318EB10(v49, 0, a2, a1);
          *(_QWORD *)v7 = &unk_4A33610;
        }
        goto LABEL_19;
      case 1:
        v51 = sub_22077B0(0x20u);
        v7 = v51;
        if ( v51 )
        {
          sub_318EB10(v51, 17, a2, a1);
          *(_QWORD *)v7 = &unk_4A33370;
        }
        goto LABEL_19;
      case 2:
        v50 = sub_22077B0(0x20u);
        v7 = v50;
        if ( v50 )
        {
          sub_318EB10(v50, 16, a2, a1);
          *(_QWORD *)v7 = &unk_4A332B0;
        }
        goto LABEL_19;
      case 3:
        v73 = sub_22077B0(0x20u);
        v7 = v73;
        if ( v73 )
        {
          sub_318EB10(v73, 15, a2, a1);
          *(_QWORD *)v7 = &unk_4A33310;
        }
        goto LABEL_19;
      case 4:
        v72 = sub_22077B0(0x20u);
        v7 = v72;
        if ( v72 )
        {
          sub_318EB10(v72, 18, a2, a1);
          *(_QWORD *)v7 = &unk_4A334F0;
        }
        break;
      case 5:
        v71 = sub_22077B0(0x20u);
        v7 = v71;
        if ( v71 )
        {
          sub_318EB10(v71, 21, a2, a1);
          *(_QWORD *)v7 = &unk_4A33490;
        }
        goto LABEL_19;
      case 6:
        v69 = sub_22077B0(0x20u);
        v7 = v69;
        if ( v69 )
        {
          sub_318EB10(v69, 22, a2, a1);
          *(_QWORD *)v7 = &unk_4A33550;
        }
        v70 = *(_QWORD *)(v4 + 8);
        *(_QWORD *)(v4 + 8) = v7;
        if ( v70 )
        {
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v70 + 8LL))(v70);
          v7 = *(_QWORD *)(v4 + 8);
        }
        sub_3188190(a1, *(_QWORD *)(a2 - 32), a2);
        return v7;
      case 7:
        v68 = sub_22077B0(0x20u);
        v7 = v68;
        if ( v68 )
        {
          sub_318EB10(v68, 19, a2, a1);
          *(_QWORD *)v7 = &unk_4A333D0;
        }
        goto LABEL_19;
      case 8:
        v67 = sub_22077B0(0x20u);
        v7 = v67;
        if ( v67 )
        {
          sub_318EB10(v67, 20, a2, a1);
          *(_QWORD *)v7 = &unk_4A33430;
        }
        goto LABEL_19;
      case 9:
        v66 = sub_22077B0(0x20u);
        v7 = v66;
        if ( v66 )
        {
          sub_318EB10(v66, 8, a2, a1);
          *(_QWORD *)v7 = &unk_4A33010;
        }
        goto LABEL_19;
      case 10:
        v65 = sub_22077B0(0x20u);
        v7 = v65;
        if ( v65 )
        {
          sub_318EB10(v65, 9, a2, a1);
          *(_QWORD *)v7 = &unk_4A33070;
        }
        goto LABEL_19;
      case 11:
        v64 = sub_22077B0(0x20u);
        v7 = v64;
        if ( v64 )
        {
          sub_318EB10(v64, 10, a2, a1);
          *(_QWORD *)v7 = &unk_4A330D0;
        }
        goto LABEL_19;
      case 12:
        v63 = sub_22077B0(0x20u);
        v7 = v63;
        if ( v63 )
        {
          sub_318EB10(v63, 13, a2, a1);
          *(_QWORD *)v7 = &unk_4A331F0;
        }
        break;
      case 13:
        v62 = sub_22077B0(0x20u);
        v7 = v62;
        if ( v62 )
        {
          sub_318EB10(v62, 14, a2, a1);
          *(_QWORD *)v7 = &unk_4A33250;
        }
        break;
      case 14:
        v56 = sub_22077B0(0x20u);
        v7 = v56;
        if ( v56 )
        {
          sub_318EB10(v56, 11, a2, a1);
          *(_QWORD *)v7 = &unk_4A33130;
        }
        v57 = *(_QWORD *)(v4 + 8);
        *(_QWORD *)(v4 + 8) = v7;
        if ( v57 )
        {
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v57 + 8LL))(v57);
          v7 = *(_QWORD *)(v4 + 8);
        }
        v78 = sub_AC31F0(a2);
        if ( !BYTE4(v78) )
        {
          v58 = (unsigned int)v78;
          if ( (_DWORD)v78 )
          {
            v59 = 0;
            do
            {
              v60 = (unsigned int)v59++;
              v61 = sub_AD6690(a2, v60);
              sub_3188190(a1, v61, a2);
            }
            while ( v58 != v59 );
          }
        }
        return v7;
      case 15:
      case 16:
      case 19:
        v10 = sub_22077B0(0x20u);
        v7 = v10;
        if ( v10 )
        {
          sub_318EB10(v10, 5, a2, a1);
          *(_QWORD *)v7 = &unk_4A32EF0;
        }
LABEL_19:
        v11 = *(_QWORD *)(v4 + 8);
        *(_QWORD *)(v4 + 8) = v7;
        if ( v11 )
        {
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v11 + 8LL))(v11);
          v7 = *(_QWORD *)(v4 + 8);
        }
        v12 = 4LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
        if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        {
          v13 = *(__int64 **)(a2 - 8);
          v14 = &v13[v12];
        }
        else
        {
          v14 = (__int64 *)a2;
          v13 = (__int64 *)(a2 - v12 * 8);
        }
        while ( v14 != v13 )
        {
          v15 = *v13;
          v13 += 4;
          sub_3188190(a1, v15, a2);
        }
        return v7;
      case 17:
        v55 = sub_22077B0(0x20u);
        v7 = v55;
        if ( v55 )
        {
          sub_318EB10(v55, 6, a2, a1);
          *(_QWORD *)v7 = &unk_4A32F50;
        }
        break;
      case 18:
        v54 = sub_22077B0(0x20u);
        v7 = v54;
        if ( v54 )
        {
          sub_318EB10(v54, 7, a2, a1);
          *(_QWORD *)v7 = &unk_4A32FB0;
        }
        break;
      case 20:
        v53 = sub_22077B0(0x20u);
        v7 = v53;
        if ( v53 )
        {
          sub_318EB10(v53, 12, a2, a1);
          *(_QWORD *)v7 = &unk_4A33190;
        }
        break;
      case 21:
        v52 = sub_22077B0(0x20u);
        v7 = v52;
        if ( v52 )
        {
          sub_318EB10(v52, 23, a2, a1);
          *(_QWORD *)v7 = &unk_4A335B0;
        }
        break;
      default:
        switch ( v5 )
        {
          case 22:
            v76 = sub_22077B0(0x20u);
            v7 = v76;
            if ( v76 )
            {
              sub_318EB10(v76, 1, a2, a1);
              *(_QWORD *)v7 = &unk_4A32EB0;
            }
            break;
          case 23:
            return sub_3186770(a1, a2);
          case 24:
          case 25:
            v77 = sub_22077B0(0x20u);
            v7 = v77;
            if ( v77 )
            {
              sub_318EB10(v77, 2, a2, a1);
              *(_QWORD *)v7 = &unk_4A32E90;
            }
            break;
          default:
LABEL_184:
            BUG();
        }
        break;
    }
  }
  else
  {
    switch ( *(_BYTE *)a2 )
    {
      case 0x1E:
        v20 = sub_22077B0(0x28u);
        v7 = v20;
        if ( v20 )
        {
          sub_318EB10(v20, 37, a2, a1);
          *(_DWORD *)(v7 + 32) = 13;
          *(_QWORD *)v7 = &unk_4A33C20;
        }
        break;
      case 0x1F:
        v46 = sub_22077B0(0x28u);
        v7 = v46;
        if ( v46 )
        {
          sub_318EB10(v46, 34, a2, a1);
          *(_DWORD *)(v7 + 32) = 10;
          *(_QWORD *)v7 = &unk_4A33910;
        }
        break;
      case 0x20:
        v47 = sub_22077B0(0x28u);
        v7 = v47;
        if ( v47 )
        {
          sub_318EB10(v47, 49, a2, a1);
          *(_DWORD *)(v7 + 32) = 25;
          *(_QWORD *)v7 = &unk_4A34160;
        }
        break;
      case 0x22:
        v42 = sub_22077B0(0x28u);
        v7 = v42;
        if ( v42 )
        {
          sub_318EB10(v42, 39, a2, a1);
          *(_DWORD *)(v7 + 32) = 15;
          *(_QWORD *)v7 = &unk_4A33D00;
        }
        break;
      case 0x23:
        v43 = sub_22077B0(0x28u);
        v7 = v43;
        if ( v43 )
        {
          sub_318EB10(v43, 47, a2, a1);
          *(_DWORD *)(v7 + 32) = 23;
          *(_QWORD *)v7 = &unk_4A340F0;
        }
        break;
      case 0x24:
        v44 = sub_22077B0(0x28u);
        v7 = v44;
        if ( v44 )
        {
          sub_318EB10(v44, 57, a2, a1);
          *(_DWORD *)(v7 + 32) = 62;
          *(_QWORD *)v7 = &unk_4A33BB0;
        }
        break;
      case 0x25:
        v45 = sub_22077B0(0x28u);
        v7 = v45;
        if ( v45 )
        {
          sub_318EB10(v45, 45, a2, a1);
          *(_DWORD *)(v7 + 32) = 21;
          *(_QWORD *)v7 = &unk_4A33FA0;
        }
        break;
      case 0x26:
        v34 = sub_22077B0(0x28u);
        v7 = v34;
        if ( v34 )
        {
          sub_318EB10(v34, 44, a2, a1);
          *(_DWORD *)(v7 + 32) = 20;
          *(_QWORD *)v7 = &unk_4A33F30;
        }
        break;
      case 0x27:
        v35 = sub_22077B0(0x28u);
        v7 = v35;
        if ( v35 )
        {
          sub_318EB10(v35, 48, a2, a1);
          *(_DWORD *)(v7 + 32) = 24;
          *(_QWORD *)v7 = &unk_4A34080;
        }
        break;
      case 0x28:
        v36 = sub_22077B0(0x28u);
        v7 = v36;
        if ( v36 )
        {
          sub_318EB10(v36, 40, a2, a1);
          *(_DWORD *)(v7 + 32) = 16;
          *(_QWORD *)v7 = &unk_4A33D70;
        }
        break;
      case 0x29:
        v37 = sub_22077B0(0x28u);
        v7 = v37;
        if ( !v37 )
          break;
        if ( *(_BYTE *)a2 != 41 )
          goto LABEL_184;
        sub_318EB10(v37, 50, a2, a1);
        *(_DWORD *)(v7 + 32) = 26;
        *(_QWORD *)v7 = &unk_4A341D0;
        break;
      case 0x2A:
      case 0x2B:
      case 0x2C:
      case 0x2D:
      case 0x2E:
      case 0x2F:
      case 0x30:
      case 0x31:
      case 0x32:
      case 0x33:
      case 0x34:
      case 0x35:
      case 0x36:
      case 0x37:
      case 0x38:
      case 0x39:
      case 0x3A:
      case 0x3B:
        v7 = sub_22077B0(0x28u);
        if ( v7 )
        {
          switch ( *(_BYTE *)a2 )
          {
            case '*':
              v75 = 27;
              goto LABEL_156;
            case '+':
              v75 = 28;
              goto LABEL_156;
            case ',':
              v75 = 29;
              goto LABEL_156;
            case '-':
              v75 = 30;
              goto LABEL_156;
            case '.':
              v75 = 31;
              goto LABEL_156;
            case '/':
              v75 = 32;
              goto LABEL_156;
            case '0':
              v75 = 33;
              goto LABEL_156;
            case '1':
              v75 = 34;
              goto LABEL_156;
            case '2':
              v75 = 35;
              goto LABEL_156;
            case '3':
              v75 = 36;
              goto LABEL_156;
            case '4':
              v75 = 37;
              goto LABEL_156;
            case '5':
              v75 = 38;
              goto LABEL_156;
            case '6':
              v75 = 39;
              goto LABEL_156;
            case '7':
              v75 = 40;
              goto LABEL_156;
            case '8':
              v75 = 41;
              goto LABEL_156;
            case '9':
              v75 = 42;
              goto LABEL_156;
            case ':':
              v75 = 43;
              goto LABEL_156;
            case ';':
              v75 = 44;
LABEL_156:
              sub_318EB10(v7, 51, a2, a1);
              *(_DWORD *)(v7 + 32) = v75;
              *(_QWORD *)v7 = &unk_4A34240;
              goto LABEL_8;
            default:
              goto LABEL_184;
          }
        }
        break;
      case 0x3C:
        v33 = sub_22077B0(0x28u);
        v7 = v33;
        if ( v33 )
        {
          sub_318EB10(v33, 54, a2, a1);
          *(_DWORD *)(v7 + 32) = 47;
          *(_QWORD *)v7 = &unk_4A34390;
        }
        break;
      case 0x3D:
        v48 = sub_22077B0(0x28u);
        v7 = v48;
        if ( v48 )
        {
          sub_318EB10(v48, 35, a2, a1);
          *(_DWORD *)(v7 + 32) = 11;
          *(_QWORD *)v7 = &unk_4A33AD0;
        }
        break;
      case 0x3E:
        v19 = sub_22077B0(0x28u);
        v7 = v19;
        if ( v19 )
        {
          sub_318EB10(v19, 36, a2, a1);
          *(_DWORD *)(v7 + 32) = 12;
          *(_QWORD *)v7 = &unk_4A33B40;
        }
        break;
      case 0x3F:
        v26 = sub_22077B0(0x28u);
        v7 = v26;
        if ( v26 )
        {
          sub_318EB10(v26, 46, a2, a1);
          *(_DWORD *)(v7 + 32) = 22;
          *(_QWORD *)v7 = &unk_4A34010;
        }
        break;
      case 0x40:
        v27 = sub_22077B0(0x28u);
        v7 = v27;
        if ( v27 )
        {
          sub_318EB10(v27, 29, a2, a1);
          *(_DWORD *)(v7 + 32) = 5;
          *(_QWORD *)v7 = &unk_4A33670;
        }
        break;
      case 0x41:
        v28 = sub_22077B0(0x28u);
        v7 = v28;
        if ( v28 )
        {
          sub_318EB10(v28, 53, a2, a1);
          *(_DWORD *)(v7 + 32) = 46;
          *(_QWORD *)v7 = &unk_4A34320;
        }
        break;
      case 0x42:
        v29 = sub_22077B0(0x28u);
        v7 = v29;
        if ( v29 )
        {
          sub_318EB10(v29, 52, a2, a1);
          *(_DWORD *)(v7 + 32) = 45;
          *(_QWORD *)v7 = &unk_4A342B0;
        }
        break;
      case 0x43:
      case 0x44:
      case 0x45:
      case 0x46:
      case 0x47:
      case 0x48:
      case 0x49:
      case 0x4A:
      case 0x4B:
      case 0x4C:
      case 0x4D:
      case 0x4E:
      case 0x4F:
        v7 = sub_22077B0(0x28u);
        if ( v7 )
        {
          switch ( *(_BYTE *)a2 )
          {
            case 'C':
              v74 = 57;
              goto LABEL_142;
            case 'D':
              v74 = 48;
              goto LABEL_142;
            case 'E':
              v74 = 49;
              goto LABEL_142;
            case 'F':
              v74 = 50;
              goto LABEL_142;
            case 'G':
              v74 = 51;
              goto LABEL_142;
            case 'H':
              v74 = 56;
              goto LABEL_142;
            case 'I':
              v74 = 55;
              goto LABEL_142;
            case 'J':
              v74 = 58;
              goto LABEL_142;
            case 'K':
              v74 = 52;
              goto LABEL_142;
            case 'L':
              v74 = 53;
              goto LABEL_142;
            case 'M':
              v74 = 54;
              goto LABEL_142;
            case 'N':
              v74 = 59;
              goto LABEL_142;
            case 'O':
              v74 = 60;
LABEL_142:
              sub_318EB10(v7, 55, a2, a1);
              *(_DWORD *)(v7 + 32) = v74;
              *(_QWORD *)v7 = &unk_4A34400;
              goto LABEL_8;
            default:
              goto LABEL_184;
          }
        }
        break;
      case 0x50:
        v32 = sub_22077B0(0x28u);
        v7 = v32;
        if ( v32 )
        {
          sub_318EB10(v32, 43, a2, a1);
          *(_DWORD *)(v7 + 32) = 19;
          *(_QWORD *)v7 = &unk_4A33EC0;
        }
        break;
      case 0x51:
        v30 = sub_22077B0(0x28u);
        v7 = v30;
        if ( v30 )
        {
          sub_318EB10(v30, 42, a2, a1);
          *(_DWORD *)(v7 + 32) = 18;
          *(_QWORD *)v7 = &unk_4A33E50;
        }
        break;
      case 0x52:
        v31 = sub_22077B0(0x28u);
        v7 = v31;
        if ( v31 )
        {
          sub_318EB10(v31, 58, a2, a1);
          *(_DWORD *)(v7 + 32) = 63;
          *(_QWORD *)v7 = &unk_4A344E0;
        }
        break;
      case 0x53:
        v16 = sub_22077B0(0x28u);
        v7 = v16;
        if ( v16 )
        {
          sub_318EB10(v16, 59, a2, a1);
          *(_DWORD *)(v7 + 32) = 64;
          *(_QWORD *)v7 = &unk_4A34550;
        }
        break;
      case 0x54:
        v17 = sub_22077B0(0x28u);
        v7 = v17;
        if ( v17 )
        {
          sub_318EB10(v17, 56, a2, a1);
          *(_DWORD *)(v7 + 32) = 61;
          *(_QWORD *)v7 = &unk_4A34470;
        }
        break;
      case 0x55:
        v18 = sub_22077B0(0x28u);
        v7 = v18;
        if ( v18 )
        {
          sub_318EB10(v18, 38, a2, a1);
          *(_DWORD *)(v7 + 32) = 14;
          *(_QWORD *)v7 = &unk_4A33C90;
        }
        break;
      case 0x56:
        v21 = sub_22077B0(0x28u);
        v7 = v21;
        if ( v21 )
        {
          sub_318EB10(v21, 33, a2, a1);
          *(_DWORD *)(v7 + 32) = 9;
          *(_QWORD *)v7 = &unk_4A336E0;
        }
        break;
      case 0x59:
        v22 = sub_22077B0(0x28u);
        v7 = v22;
        if ( v22 )
        {
          sub_318EB10(v22, 27, a2, a1);
          *(_DWORD *)(v7 + 32) = 3;
          *(_QWORD *)v7 = &unk_4A339F0;
        }
        break;
      case 0x5A:
        v23 = sub_22077B0(0x28u);
        v7 = v23;
        if ( v23 )
        {
          sub_318EB10(v23, 25, a2, a1);
          *(_DWORD *)(v7 + 32) = 1;
          *(_QWORD *)v7 = &unk_4A337C0;
        }
        break;
      case 0x5B:
        v24 = sub_22077B0(0x28u);
        v7 = v24;
        if ( v24 )
        {
          sub_318EB10(v24, 26, a2, a1);
          *(_DWORD *)(v7 + 32) = 2;
          *(_QWORD *)v7 = &unk_4A33750;
        }
        break;
      case 0x5C:
        v25 = sub_22077B0(0x28u);
        v7 = v25;
        if ( v25 )
        {
          sub_318EB10(v25, 30, a2, a1);
          *(_DWORD *)(v7 + 32) = 6;
          *(_QWORD *)v7 = &unk_4A33830;
        }
        break;
      case 0x5D:
        v38 = sub_22077B0(0x28u);
        v7 = v38;
        if ( v38 )
        {
          sub_318EB10(v38, 31, a2, a1);
          *(_DWORD *)(v7 + 32) = 7;
          *(_QWORD *)v7 = &unk_4A33980;
        }
        break;
      case 0x5E:
        v39 = sub_22077B0(0x28u);
        v7 = v39;
        if ( v39 )
        {
          sub_318EB10(v39, 32, a2, a1);
          *(_DWORD *)(v7 + 32) = 8;
          *(_QWORD *)v7 = &unk_4A338A0;
        }
        break;
      case 0x5F:
        v40 = sub_22077B0(0x28u);
        v7 = v40;
        if ( v40 )
        {
          sub_318EB10(v40, 41, a2, a1);
          *(_DWORD *)(v7 + 32) = 17;
          *(_QWORD *)v7 = &unk_4A33DE0;
        }
        break;
      case 0x60:
        v41 = sub_22077B0(0x28u);
        v7 = v41;
        if ( v41 )
        {
          sub_318EB10(v41, 28, a2, a1);
          *(_DWORD *)(v7 + 32) = 4;
          *(_QWORD *)v7 = &unk_4A33A60;
        }
        break;
      default:
        v6 = sub_22077B0(0x28u);
        v7 = v6;
        if ( v6 )
        {
          sub_318EB10(v6, 24, a2, a1);
          *(_DWORD *)(v7 + 32) = 0;
          *(_QWORD *)v7 = &unk_4A345C0;
        }
        break;
    }
  }
LABEL_8:
  v8 = *(_QWORD *)(v4 + 8);
  *(_QWORD *)(v4 + 8) = v7;
  if ( v8 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v8 + 8LL))(v8);
    return *(_QWORD *)(v4 + 8);
  }
  return v7;
}
