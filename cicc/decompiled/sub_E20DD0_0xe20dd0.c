// Function: sub_E20DD0
// Address: 0xe20dd0
//
size_t __fastcall sub_E20DD0(__int64 *a1, unsigned int a2)
{
  size_t result; // rax
  unsigned __int64 v4; // rdx
  char *v5; // rdi
  unsigned __int64 v6; // rdx
  __int64 v7; // rax
  char *v8; // rdx
  int v9; // edi
  char v10; // al
  char v11; // al
  char *v12; // r13
  size_t v13; // r12
  char *v14; // rcx
  size_t v15; // rdx
  char *v16; // rdi
  char *v17; // rax
  unsigned __int64 v18; // rdx
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  char *v21; // rdi
  unsigned __int64 v22; // rdx
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  char *v25; // rdi
  unsigned __int64 v26; // rdx
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  char *v29; // rdi
  unsigned __int64 v30; // rdx
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  char *v33; // rdi
  unsigned __int64 v34; // rdx
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  char *v37; // rdi
  unsigned __int64 v38; // rdx
  __int64 v39; // rax
  unsigned __int64 v40; // rdx
  char *v41; // rdi
  unsigned __int64 v42; // rdx
  __int64 v43; // rax
  unsigned __int64 v44; // rdx
  char *v45; // rdi
  unsigned __int64 v46; // rdx
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  char *v49; // rdi
  unsigned __int64 v50; // rdx
  __int64 v51; // rax
  unsigned __int64 v52; // rdx
  char *v53; // rdi
  unsigned __int64 v54; // rdx
  __int64 v55; // rax
  unsigned __int64 v56; // rdx
  char *v57; // rdi
  unsigned __int64 v58; // rdx
  __int64 v59; // rax
  unsigned __int64 v60; // rdx
  char *v61; // rdi
  size_t v62; // rcx
  unsigned __int64 v63; // rdx
  __int64 v64; // rax
  char s[16]; // [rsp+0h] [rbp-40h] BYREF
  char v66; // [rsp+10h] [rbp-30h]

  switch ( a2 )
  {
    case 0u:
      result = a1[1];
      v28 = a1[2];
      v29 = (char *)*a1;
      if ( result + 2 <= v28 )
        goto LABEL_42;
      v30 = 2 * v28;
      if ( result + 994 > v30 )
        a1[2] = result + 994;
      else
        a1[2] = v30;
      v31 = realloc(v29);
      *a1 = v31;
      v29 = (char *)v31;
      if ( !v31 )
        goto LABEL_104;
      result = a1[1];
LABEL_42:
      *(_WORD *)&v29[result] = 12380;
      a1[1] += 2;
      return result;
    case 1u:
    case 2u:
    case 3u:
    case 4u:
    case 5u:
    case 6u:
    case 0xEu:
    case 0xFu:
    case 0x10u:
    case 0x11u:
    case 0x12u:
    case 0x13u:
    case 0x14u:
    case 0x15u:
    case 0x16u:
    case 0x17u:
    case 0x18u:
    case 0x19u:
    case 0x1Au:
    case 0x1Bu:
    case 0x1Cu:
    case 0x1Du:
    case 0x1Eu:
    case 0x1Fu:
    case 0x20u:
    case 0x21u:
    case 0x23u:
    case 0x24u:
    case 0x25u:
    case 0x26u:
      goto LABEL_10;
    case 7u:
      result = a1[1];
      v32 = a1[2];
      v33 = (char *)*a1;
      if ( result + 2 <= v32 )
        goto LABEL_48;
      v34 = 2 * v32;
      if ( result + 994 > v34 )
        a1[2] = result + 994;
      else
        a1[2] = v34;
      v35 = realloc(v33);
      *a1 = v35;
      v33 = (char *)v35;
      if ( !v35 )
        goto LABEL_104;
      result = a1[1];
LABEL_48:
      *(_WORD *)&v33[result] = 24924;
      a1[1] += 2;
      return result;
    case 8u:
      result = a1[1];
      v36 = a1[2];
      v37 = (char *)*a1;
      if ( result + 2 <= v36 )
        goto LABEL_54;
      v38 = 2 * v36;
      if ( result + 994 > v38 )
        a1[2] = result + 994;
      else
        a1[2] = v38;
      v39 = realloc(v37);
      *a1 = v39;
      v37 = (char *)v39;
      if ( !v39 )
        goto LABEL_104;
      result = a1[1];
LABEL_54:
      *(_WORD *)&v37[result] = 25180;
      a1[1] += 2;
      return result;
    case 9u:
      result = a1[1];
      v40 = a1[2];
      v41 = (char *)*a1;
      if ( result + 2 <= v40 )
        goto LABEL_60;
      v42 = 2 * v40;
      if ( result + 994 > v42 )
        a1[2] = result + 994;
      else
        a1[2] = v42;
      v43 = realloc(v41);
      *a1 = v43;
      v41 = (char *)v43;
      if ( !v43 )
        goto LABEL_104;
      result = a1[1];
LABEL_60:
      *(_WORD *)&v41[result] = 29788;
      a1[1] += 2;
      return result;
    case 0xAu:
      result = a1[1];
      v44 = a1[2];
      v45 = (char *)*a1;
      if ( result + 2 <= v44 )
        goto LABEL_66;
      v46 = 2 * v44;
      if ( result + 994 > v46 )
        a1[2] = result + 994;
      else
        a1[2] = v46;
      v47 = realloc(v45);
      *a1 = v47;
      v45 = (char *)v47;
      if ( !v47 )
        goto LABEL_104;
      result = a1[1];
LABEL_66:
      *(_WORD *)&v45[result] = 28252;
      a1[1] += 2;
      return result;
    case 0xBu:
      result = a1[1];
      v48 = a1[2];
      v49 = (char *)*a1;
      if ( result + 2 <= v48 )
        goto LABEL_72;
      v50 = 2 * v48;
      if ( result + 994 > v50 )
        a1[2] = result + 994;
      else
        a1[2] = v50;
      v51 = realloc(v49);
      *a1 = v51;
      v49 = (char *)v51;
      if ( !v51 )
        goto LABEL_104;
      result = a1[1];
LABEL_72:
      *(_WORD *)&v49[result] = 30300;
      a1[1] += 2;
      return result;
    case 0xCu:
      result = a1[1];
      v52 = a1[2];
      v53 = (char *)*a1;
      if ( result + 2 <= v52 )
        goto LABEL_78;
      v54 = 2 * v52;
      if ( result + 994 > v54 )
        a1[2] = result + 994;
      else
        a1[2] = v54;
      v55 = realloc(v53);
      *a1 = v55;
      v53 = (char *)v55;
      if ( !v55 )
        goto LABEL_104;
      result = a1[1];
LABEL_78:
      *(_WORD *)&v53[result] = 26204;
      a1[1] += 2;
      return result;
    case 0xDu:
      result = a1[1];
      v56 = a1[2];
      v57 = (char *)*a1;
      if ( result + 2 <= v56 )
        goto LABEL_84;
      v58 = 2 * v56;
      if ( result + 994 > v58 )
        a1[2] = result + 994;
      else
        a1[2] = v58;
      v59 = realloc(v57);
      *a1 = v59;
      v57 = (char *)v59;
      if ( !v59 )
        goto LABEL_104;
      result = a1[1];
LABEL_84:
      *(_WORD *)&v57[result] = 29276;
      a1[1] += 2;
      return result;
    case 0x22u:
      result = a1[1];
      v20 = a1[2];
      v21 = (char *)*a1;
      if ( result + 2 <= v20 )
        goto LABEL_30;
      v22 = 2 * v20;
      if ( result + 994 > v22 )
        a1[2] = result + 994;
      else
        a1[2] = v22;
      v23 = realloc(v21);
      *a1 = v23;
      v21 = (char *)v23;
      if ( !v23 )
        goto LABEL_104;
      result = a1[1];
LABEL_30:
      *(_WORD *)&v21[result] = 8796;
      a1[1] += 2;
      return result;
    case 0x27u:
      result = a1[1];
      v24 = a1[2];
      v25 = (char *)*a1;
      if ( result + 2 <= v24 )
        goto LABEL_36;
      v26 = 2 * v24;
      if ( result + 994 > v26 )
        a1[2] = result + 994;
      else
        a1[2] = v26;
      v27 = realloc(v25);
      *a1 = v27;
      v25 = (char *)v27;
      if ( !v27 )
        goto LABEL_104;
      result = a1[1];
LABEL_36:
      *(_WORD *)&v25[result] = 10076;
      a1[1] += 2;
      return result;
    default:
      if ( a2 == 92 )
      {
        result = a1[1];
        v4 = a1[2];
        v5 = (char *)*a1;
        if ( result + 2 <= v4 )
        {
LABEL_8:
          *(_WORD *)&v5[result] = 23644;
          a1[1] += 2;
          return result;
        }
        v6 = 2 * v4;
        if ( result + 994 > v6 )
          a1[2] = result + 994;
        else
          a1[2] = v6;
        v7 = realloc(v5);
        *a1 = v7;
        v5 = (char *)v7;
        if ( v7 )
        {
          result = a1[1];
          goto LABEL_8;
        }
LABEL_104:
        abort();
      }
LABEL_10:
      if ( a2 - 32 <= 0x5E )
      {
        result = a1[1];
        v60 = a1[2];
        v61 = (char *)*a1;
        v62 = result + 1;
        if ( result + 1 > v60 )
        {
          v63 = 2 * v60;
          if ( result + 993 > v63 )
            a1[2] = result + 993;
          else
            a1[2] = v63;
          v64 = realloc(v61);
          *a1 = v64;
          v61 = (char *)v64;
          if ( !v64 )
            goto LABEL_104;
          result = a1[1];
          v62 = result + 1;
        }
        a1[1] = v62;
        v61[result] = a2;
        return result;
      }
      v66 = 0;
      v8 = &s[14];
      v9 = 15;
      *(_OWORD *)s = 0;
      while ( 1 )
      {
        v10 = (a2 & 0xF) + 48;
        if ( (a2 & 0xF) >= 0xA )
          v10 = (a2 & 0xF) + 55;
        v8[1] = v10;
        v11 = ((unsigned __int8)a2 >> 4) + 55;
        if ( (unsigned __int8)((unsigned __int8)a2 >> 4) < 0xAu )
          v11 = ((unsigned __int8)a2 >> 4) + 48;
        v8 -= 2;
        v8[2] = v11;
        a2 >>= 8;
        if ( !a2 )
          break;
        v9 -= 2;
      }
      s[v9 - 2] = 120;
      v12 = &s[v9 - 3];
      *v12 = 92;
      result = strlen(v12);
      v13 = result;
      if ( result )
      {
        v14 = (char *)a1[1];
        v15 = a1[2];
        v16 = (char *)*a1;
        v17 = &v14[result];
        if ( (unsigned __int64)v17 > v15 )
        {
          v18 = 2 * v15;
          if ( (unsigned __int64)(v17 + 992) > v18 )
            a1[2] = (__int64)(v17 + 992);
          else
            a1[2] = v18;
          v19 = realloc(v16);
          *a1 = v19;
          v16 = (char *)v19;
          if ( !v19 )
            goto LABEL_104;
          v14 = (char *)a1[1];
        }
        result = (size_t)memcpy(&v16[(_QWORD)v14], v12, v13);
        a1[1] += v13;
      }
      return result;
  }
}
