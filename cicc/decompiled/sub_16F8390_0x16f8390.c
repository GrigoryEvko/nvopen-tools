// Function: sub_16F8390
// Address: 0x16f8390
//
const char *__fastcall sub_16F8390(__int64 a1, char *a2, unsigned __int64 a3, unsigned __int64 a4, _DWORD *a5, int a6)
{
  __int64 v6; // rax
  unsigned __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // r8
  int v13; // r9d
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  char *v16; // rbx
  _BOOL8 v17; // rcx
  unsigned __int64 v18; // rdx
  char *v19; // rbx
  __int64 v20; // rax
  unsigned __int64 v21; // rax
  char *v22; // rdx
  char v23; // cl
  const char *result; // rax
  char v25; // al
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
  __int64 v36; // rdx
  __int64 v37; // rcx
  int v38; // r8d
  int v39; // r9d
  char v40; // bl
  unsigned int v41; // edi
  unsigned __int64 v42; // rbx
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rcx
  int v46; // r8d
  int v47; // r9d
  char v48; // bl
  unsigned int v49; // edi
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rcx
  int v61; // r8d
  int v62; // r9d
  char v63; // bl
  unsigned int v64; // edi
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rax
  char *v71; // [rsp+10h] [rbp-90h] BYREF
  unsigned __int64 v72; // [rsp+18h] [rbp-88h]
  const char *v73; // [rsp+20h] [rbp-80h] BYREF
  char v74; // [rsp+30h] [rbp-70h]
  char v75; // [rsp+31h] [rbp-6Fh]
  unsigned __int64 v76[3]; // [rsp+40h] [rbp-60h] BYREF
  _QWORD *v77; // [rsp+58h] [rbp-48h]
  __int64 v78; // [rsp+60h] [rbp-40h]
  _QWORD v79[7]; // [rsp+68h] [rbp-38h] BYREF

  v6 = 0;
  v10 = (unsigned int)a5[3];
  v71 = a2;
  v11 = 0;
  v72 = a3;
  a5[2] = 0;
  if ( a3 > v10 )
  {
    sub_16CD150((__int64)a5, a5 + 4, a3, 1, (int)a5, a6);
    v6 = (unsigned int)a5[2];
    v11 = v6;
  }
  if ( a4 != -1 )
  {
    while ( 1 )
    {
      sub_16F64E0((__int64)a5, (char *)(*(_QWORD *)a5 + v11), v71, &v71[a4]);
      v14 = v72;
      if ( v72 < a4 )
      {
        v72 = 0;
        v71 += v14;
        v16 = v71;
        if ( *v71 == 10 || *v71 == 13 )
        {
LABEL_12:
          v20 = (unsigned int)a5[2];
          if ( (unsigned int)v20 >= a5[3] )
          {
            sub_16CD150((__int64)a5, a5 + 4, 0, 1, v12, v13);
            v20 = (unsigned int)a5[2];
          }
          *(_BYTE *)(*(_QWORD *)a5 + v20) = 10;
          v21 = v72;
          ++a5[2];
          v22 = v71;
          if ( v21 <= 1 )
          {
            if ( v21 )
            {
              v21 = 1;
              goto LABEL_17;
            }
          }
          else
          {
            v23 = v71[1];
            if ( v23 == 13 || v23 == 10 )
            {
              --v21;
              v22 = v71 + 1;
            }
LABEL_17:
            --v21;
            ++v22;
          }
          v71 = v22;
          v72 = v21;
          goto LABEL_19;
        }
        v17 = 0;
        v18 = 0;
        goto LABEL_9;
      }
      v15 = v72 - a4;
      v16 = &v71[a4];
      v71 = v16;
      if ( v15 == -1 )
        break;
      v72 = v15;
      if ( *v16 == 10 || *v16 == 13 )
        goto LABEL_12;
      if ( v15 != 1 )
      {
        v17 = v15 != 0;
        v18 = v15 - v17;
LABEL_9:
        v19 = &v16[v17];
        v72 = v18;
        v71 = v19;
        switch ( *v19 )
        {
          case 9:
          case 116:
            v29 = (unsigned int)a5[2];
            if ( (unsigned int)v29 >= a5[3] )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v12, v13);
              v29 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v29) = 9;
            v18 = v72;
            ++a5[2];
            v19 = v71;
            goto LABEL_40;
          case 10:
          case 13:
            if ( v18 <= 1 )
              goto LABEL_40;
            v25 = v19[1];
            if ( v25 == 13 || v25 == 10 )
            {
              --v18;
              ++v19;
            }
            goto LABEL_41;
          case 32:
            v31 = (unsigned int)a5[2];
            if ( (unsigned int)v31 >= a5[3] )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v12, v13);
              v31 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v31) = 32;
            v18 = v72;
            ++a5[2];
            v19 = v71;
            goto LABEL_40;
          case 34:
            v30 = (unsigned int)a5[2];
            if ( (unsigned int)v30 >= a5[3] )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v12, v13);
              v30 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v30) = 34;
            v18 = v72;
            ++a5[2];
            v19 = v71;
            goto LABEL_40;
          case 47:
            v35 = (unsigned int)a5[2];
            if ( (unsigned int)v35 >= a5[3] )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v12, v13);
              v35 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v35) = 47;
            v18 = v72;
            ++a5[2];
            v19 = v71;
            goto LABEL_40;
          case 48:
            v34 = (unsigned int)a5[2];
            if ( (unsigned int)v34 >= a5[3] )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v12, v13);
              v34 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v34) = 0;
            v18 = v72;
            ++a5[2];
            v19 = v71;
            goto LABEL_40;
          case 76:
            v26 = (unsigned int)a5[2];
            if ( (unsigned int)v26 >= a5[3] )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v12, v13);
              v26 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v26) = -30;
            v27 = (unsigned int)(a5[2] + 1);
            a5[2] = v27;
            if ( a5[3] <= (unsigned int)v27 )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v12, v13);
              v27 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v27) = 0x80;
            v28 = (unsigned int)(a5[2] + 1);
            a5[2] = v28;
            if ( a5[3] <= (unsigned int)v28 )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v12, v13);
              v28 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v28) = -88;
            v18 = v72;
            ++a5[2];
            v19 = v71;
            goto LABEL_40;
          case 78:
            v32 = (unsigned int)a5[2];
            if ( (unsigned int)v32 >= a5[3] )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v12, v13);
              v32 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v32) = -62;
            v33 = (unsigned int)(a5[2] + 1);
            a5[2] = v33;
            if ( a5[3] <= (unsigned int)v33 )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v12, v13);
              v33 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v33) = -123;
            v18 = v72;
            ++a5[2];
            v19 = v71;
            goto LABEL_40;
          case 80:
            v65 = (unsigned int)a5[2];
            if ( (unsigned int)v65 >= a5[3] )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v12, v13);
              v65 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v65) = -30;
            v66 = (unsigned int)(a5[2] + 1);
            a5[2] = v66;
            if ( a5[3] <= (unsigned int)v66 )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v12, v13);
              v66 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v66) = 0x80;
            v67 = (unsigned int)(a5[2] + 1);
            a5[2] = v67;
            if ( a5[3] <= (unsigned int)v67 )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v12, v13);
              v67 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v67) = -87;
            v18 = v72;
            ++a5[2];
            v19 = v71;
            goto LABEL_40;
          case 85:
            if ( v18 <= 8 )
              goto LABEL_40;
            if ( sub_16D2B80((__int64)(v19 + 1), 8, 0x10u, v76)
              || (v63 = v76[0], v64 = v76[0], v76[0] != LODWORD(v76[0])) )
            {
              v64 = 65533;
LABEL_110:
              sub_16F69C0(v64, (__int64)a5, v59, v60, v61, v62);
              goto LABEL_111;
            }
            if ( LODWORD(v76[0]) > 0x7F )
              goto LABEL_110;
            v70 = (unsigned int)a5[2];
            if ( (unsigned int)v70 >= a5[3] )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v61, v62);
              v70 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v70) = v63;
            ++a5[2];
LABEL_111:
            v42 = v72;
            if ( v72 <= 7 )
            {
LABEL_132:
              v19 = &v71[v42];
              v18 = 0;
            }
            else
            {
              v18 = v72 - 8;
              v19 = v71 + 8;
LABEL_40:
              if ( v18 )
              {
LABEL_41:
                --v18;
                ++v19;
              }
            }
            v71 = v19;
            v72 = v18;
            break;
          case 92:
            v58 = (unsigned int)a5[2];
            if ( (unsigned int)v58 >= a5[3] )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v12, v13);
              v58 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v58) = 92;
            v18 = v72;
            ++a5[2];
            v19 = v71;
            goto LABEL_40;
          case 95:
            v56 = (unsigned int)a5[2];
            if ( (unsigned int)v56 >= a5[3] )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v12, v13);
              v56 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v56) = -62;
            v57 = (unsigned int)(a5[2] + 1);
            a5[2] = v57;
            if ( a5[3] <= (unsigned int)v57 )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v12, v13);
              v57 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v57) = -96;
            v18 = v72;
            ++a5[2];
            v19 = v71;
            goto LABEL_40;
          case 97:
            v55 = (unsigned int)a5[2];
            if ( (unsigned int)v55 >= a5[3] )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v12, v13);
              v55 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v55) = 7;
            v18 = v72;
            ++a5[2];
            v19 = v71;
            goto LABEL_40;
          case 98:
            v54 = (unsigned int)a5[2];
            if ( (unsigned int)v54 >= a5[3] )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v12, v13);
              v54 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v54) = 8;
            v18 = v72;
            ++a5[2];
            v19 = v71;
            goto LABEL_40;
          case 101:
            v53 = (unsigned int)a5[2];
            if ( (unsigned int)v53 >= a5[3] )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v12, v13);
              v53 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v53) = 27;
            v18 = v72;
            ++a5[2];
            v19 = v71;
            goto LABEL_40;
          case 102:
            v52 = (unsigned int)a5[2];
            if ( (unsigned int)v52 >= a5[3] )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v12, v13);
              v52 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v52) = 12;
            v18 = v72;
            ++a5[2];
            v19 = v71;
            goto LABEL_40;
          case 110:
            v51 = (unsigned int)a5[2];
            if ( (unsigned int)v51 >= a5[3] )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v12, v13);
              v51 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v51) = 10;
            v18 = v72;
            ++a5[2];
            v19 = v71;
            goto LABEL_40;
          case 114:
            v50 = (unsigned int)a5[2];
            if ( (unsigned int)v50 >= a5[3] )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v12, v13);
              v50 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v50) = 13;
            v18 = v72;
            ++a5[2];
            v19 = v71;
            goto LABEL_40;
          case 117:
            if ( v18 <= 4 )
              goto LABEL_40;
            if ( sub_16D2B80((__int64)(v19 + 1), 4, 0x10u, v76)
              || (v48 = v76[0], v49 = v76[0], v76[0] != LODWORD(v76[0])) )
            {
              v49 = 65533;
LABEL_77:
              sub_16F69C0(v49, (__int64)a5, v44, v45, v46, v47);
              goto LABEL_78;
            }
            if ( LODWORD(v76[0]) > 0x7F )
              goto LABEL_77;
            v68 = (unsigned int)a5[2];
            if ( (unsigned int)v68 >= a5[3] )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v46, v47);
              v68 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v68) = v48;
            ++a5[2];
LABEL_78:
            v42 = v72;
            if ( v72 <= 3 )
              goto LABEL_132;
            v18 = v72 - 4;
            v19 = v71 + 4;
            goto LABEL_40;
          case 118:
            v43 = (unsigned int)a5[2];
            if ( (unsigned int)v43 >= a5[3] )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v12, v13);
              v43 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v43) = 11;
            v18 = v72;
            ++a5[2];
            v19 = v71;
            goto LABEL_40;
          case 120:
            if ( v18 <= 2 )
              goto LABEL_40;
            if ( sub_16D2B80((__int64)(v19 + 1), 2, 0x10u, v76)
              || (v40 = v76[0], v41 = v76[0], v76[0] != LODWORD(v76[0])) )
            {
              v41 = 65533;
LABEL_67:
              sub_16F69C0(v41, (__int64)a5, v36, v37, v38, v39);
              goto LABEL_68;
            }
            if ( LODWORD(v76[0]) > 0x7F )
              goto LABEL_67;
            v69 = (unsigned int)a5[2];
            if ( (unsigned int)v69 >= a5[3] )
            {
              sub_16CD150((__int64)a5, a5 + 4, 0, 1, v38, v39);
              v69 = (unsigned int)a5[2];
            }
            *(_BYTE *)(*(_QWORD *)a5 + v69) = v40;
            ++a5[2];
LABEL_68:
            v42 = v72;
            if ( v72 <= 1 )
              goto LABEL_132;
            v18 = v72 - 2;
            v19 = v71 + 2;
            goto LABEL_40;
          default:
            LOBYTE(v79[0]) = 0;
            v77 = v79;
            v73 = "Unrecognized escape code!";
            LODWORD(v76[0]) = 0;
            v78 = 0;
            v76[1] = (unsigned __int64)v19;
            v76[2] = 1;
            v75 = 1;
            v74 = 3;
            sub_16F8380(a1, (__int64)&v73, (__int64)v76, v17, v12);
            result = byte_3F871B3;
            if ( v77 != v79 )
            {
              j_j___libc_free_0(v77, v79[0] + 1LL);
              return byte_3F871B3;
            }
            return result;
        }
      }
LABEL_19:
      a4 = sub_16D23E0(&v71, "\\\r\n", 3, 0);
      if ( a4 == -1 )
      {
        v6 = (unsigned int)a5[2];
        goto LABEL_25;
      }
      v11 = (unsigned int)a5[2];
    }
    v72 = -1;
    if ( *v16 == 10 || *v16 == 13 )
      goto LABEL_12;
    v17 = 1;
    v18 = -2;
    goto LABEL_9;
  }
LABEL_25:
  sub_16F64E0((__int64)a5, (char *)(*(_QWORD *)a5 + v6), v71, &v71[v72]);
  return *(const char **)a5;
}
