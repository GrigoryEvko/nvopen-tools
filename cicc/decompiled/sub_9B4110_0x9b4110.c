// Function: sub_9B4110
// Address: 0x9b4110
//
__int64 __fastcall sub_9B4110(unsigned __int8 *a1, __int64 a2, unsigned int a3, __m128i *a4, __int64 a5)
{
  unsigned int v8; // r15d
  bool v9; // al
  unsigned int v10; // r15d
  unsigned __int8 *v12; // rdx
  unsigned int v13; // eax
  unsigned __int64 v14; // r15
  unsigned __int64 *v15; // r15
  bool v16; // zf
  unsigned __int64 v17; // rax
  __int64 result; // rax
  int v19; // eax
  __int64 v20; // rsi
  int v21; // eax
  int v22; // eax
  bool v23; // dl
  unsigned int v24; // eax
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // rdx
  unsigned int v27; // eax
  int v28; // eax
  int v29; // eax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  unsigned __int64 v33; // rax
  unsigned int v34; // eax
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // rdx
  unsigned int v37; // eax
  unsigned __int64 v38; // rax
  int v39; // eax
  int v40; // eax
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 *v47; // rbx
  unsigned __int64 *v48; // [rsp+0h] [rbp-100h]
  unsigned int v49; // [rsp+8h] [rbp-F8h]
  bool v50; // [rsp+8h] [rbp-F8h]
  unsigned int v51; // [rsp+10h] [rbp-F0h]
  unsigned int v52; // [rsp+1Ch] [rbp-E4h]
  unsigned __int64 v53; // [rsp+20h] [rbp-E0h]
  unsigned __int8 v55; // [rsp+28h] [rbp-D8h]
  unsigned __int8 v56; // [rsp+28h] [rbp-D8h]
  unsigned __int8 v57; // [rsp+28h] [rbp-D8h]
  unsigned __int8 v58; // [rsp+28h] [rbp-D8h]
  unsigned __int8 v59; // [rsp+28h] [rbp-D8h]
  unsigned __int8 v60; // [rsp+28h] [rbp-D8h]
  unsigned __int8 v61; // [rsp+28h] [rbp-D8h]
  unsigned __int8 v62; // [rsp+28h] [rbp-D8h]
  unsigned __int8 v63; // [rsp+28h] [rbp-D8h]
  unsigned __int8 v64; // [rsp+28h] [rbp-D8h]
  unsigned __int64 *v65; // [rsp+30h] [rbp-D0h] BYREF
  unsigned int v66; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v67; // [rsp+40h] [rbp-C0h] BYREF
  unsigned int v68; // [rsp+48h] [rbp-B8h]
  unsigned __int64 v69; // [rsp+50h] [rbp-B0h] BYREF
  unsigned int v70; // [rsp+58h] [rbp-A8h]
  unsigned __int64 *v71; // [rsp+60h] [rbp-A0h] BYREF
  unsigned int v72; // [rsp+68h] [rbp-98h]
  unsigned __int64 v73; // [rsp+70h] [rbp-90h] BYREF
  unsigned int v74; // [rsp+78h] [rbp-88h]
  unsigned __int64 v75; // [rsp+80h] [rbp-80h] BYREF
  unsigned int v76; // [rsp+88h] [rbp-78h]
  unsigned __int64 v77; // [rsp+90h] [rbp-70h] BYREF
  unsigned int v78; // [rsp+98h] [rbp-68h]
  unsigned __int64 *v79; // [rsp+A0h] [rbp-60h] BYREF
  unsigned int v80; // [rsp+A8h] [rbp-58h]
  _QWORD *v81; // [rsp+B0h] [rbp-50h] BYREF
  unsigned int v82; // [rsp+B8h] [rbp-48h]
  __int64 v83; // [rsp+C0h] [rbp-40h]
  unsigned int v84; // [rsp+C8h] [rbp-38h]

  v8 = *(_DWORD *)(a5 + 8);
  if ( v8 <= 0x40 )
    v9 = *(_QWORD *)a5 == 0;
  else
    v9 = v8 == (unsigned int)sub_C444A0(a5);
  if ( v9 )
  {
    v10 = *(_DWORD *)(a5 + 24);
    if ( v10 <= 0x40 ? *(_QWORD *)(a5 + 16) == 0 : v10 == (unsigned int)sub_C444A0(a5 + 16) )
      return 0;
  }
  if ( (a1[7] & 0x40) != 0 )
    v12 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
  else
    v12 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
  sub_9B0110((__int64)&v81, *((_QWORD *)v12 + 4), a2, a3, a4);
  v13 = v82;
  v80 = v82;
  if ( v82 <= 0x40 )
  {
    v14 = (unsigned __int64)v81;
LABEL_11:
    v66 = v13;
    v15 = (unsigned __int64 *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v13) & ~v14);
    v16 = v13 == 0;
    v17 = *(unsigned int *)(a5 + 8);
    if ( v16 )
      v15 = 0;
    v52 = *(_DWORD *)(a5 + 8);
    v65 = v15;
    v53 = v17;
    goto LABEL_14;
  }
  sub_C43780(&v79, &v81);
  v13 = v80;
  if ( v80 <= 0x40 )
  {
    v14 = (unsigned __int64)v79;
    goto LABEL_11;
  }
  sub_C43D10(&v79, &v81, v30, v31, v32);
  v33 = *(unsigned int *)(a5 + 8);
  v15 = v79;
  v66 = v80;
  v65 = v79;
  v52 = v33;
  v53 = v33;
  v51 = v80;
  if ( v80 > 0x40 )
  {
    if ( v51 - (unsigned int)sub_C444A0(&v65) > 0x40 || v53 <= *v15 )
    {
      result = 0;
LABEL_78:
      if ( v65 )
      {
        v64 = result;
        j_j___libc_free_0_0(v65);
        result = v64;
      }
      goto LABEL_65;
    }
    v19 = *a1;
    v20 = a5 + 16;
    if ( (unsigned __int8)v19 > 0x1Cu )
      goto LABEL_16;
    goto LABEL_85;
  }
LABEL_14:
  result = 0;
  if ( v53 <= (unsigned __int64)v15 )
    goto LABEL_65;
  v19 = *a1;
  v20 = a5 + 16;
  if ( (unsigned __int8)v19 > 0x1Cu )
  {
LABEL_16:
    v21 = v19 - 29;
    goto LABEL_17;
  }
LABEL_85:
  v21 = *((unsigned __int16 *)a1 + 1);
LABEL_17:
  switch ( v21 )
  {
    case 26:
      v80 = *(_DWORD *)(a5 + 24);
      if ( v80 > 0x40 )
        sub_C43780(&v79, v20);
      else
        v79 = *(unsigned __int64 **)(a5 + 16);
      sub_C48380(&v79, &v65);
      break;
    case 27:
      v80 = *(_DWORD *)(a5 + 24);
      if ( v80 > 0x40 )
        sub_C43780(&v79, v20);
      else
        v79 = *(unsigned __int64 **)(a5 + 16);
      sub_C44D10(&v79, &v65);
      break;
    case 25:
      v80 = *(_DWORD *)(a5 + 24);
      if ( v80 > 0x40 )
        sub_C43780(&v79, v20);
      else
        v79 = *(unsigned __int64 **)(a5 + 16);
      sub_C47AC0(&v79, &v65);
      break;
    default:
      goto LABEL_136;
  }
  if ( v80 <= 0x40 )
  {
    v23 = v79 == 0;
  }
  else
  {
    v48 = v79;
    v49 = v80;
    v22 = sub_C444A0(&v79);
    v23 = v49 == v22;
    if ( v48 )
    {
      v50 = v49 == v22;
      j_j___libc_free_0_0(v48);
      v23 = v50;
    }
  }
  result = 1;
  if ( v23 )
  {
    v24 = v66;
    v68 = v66;
    if ( v66 > 0x40 )
    {
      sub_C43780(&v67, &v65);
      v24 = v68;
      if ( v68 > 0x40 )
      {
        sub_C43D10(&v67, &v65, v44, v45, v46);
        goto LABEL_31;
      }
      v25 = v67;
    }
    else
    {
      v25 = (unsigned __int64)v65;
    }
    v26 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v24) & ~v25;
    if ( !v24 )
      v26 = 0;
    v67 = v26;
LABEL_31:
    sub_C46250(&v67);
    sub_C46A40(&v67, v53);
    v27 = v68;
    v68 = 0;
    v70 = v27;
    v69 = v67;
    v28 = *a1;
    if ( (unsigned __int8)v28 <= 0x1Cu )
      v29 = *((unsigned __int16 *)a1 + 1);
    else
      v29 = v28 - 29;
    if ( v29 == 25 )
    {
      v72 = *(_DWORD *)(a5 + 8);
      if ( v72 > 0x40 )
        sub_C43780(&v71, a5);
      else
        v71 = *(unsigned __int64 **)a5;
      sub_C48380(&v71, &v69);
    }
    else
    {
      if ( (unsigned int)(v29 - 26) > 1 )
        goto LABEL_136;
      v72 = *(_DWORD *)(a5 + 8);
      if ( v72 > 0x40 )
        sub_C43780(&v71, a5);
      else
        v71 = *(unsigned __int64 **)a5;
      sub_C47AC0(&v71, &v69);
    }
    v34 = v66;
    v76 = v66;
    if ( v66 > 0x40 )
    {
      sub_C43780(&v75, &v65);
      v34 = v76;
      if ( v76 > 0x40 )
      {
        sub_C43D10(&v75, &v65, v41, v42, v43);
LABEL_99:
        sub_C46250(&v75);
        sub_C46A40(&v75, v53);
        v37 = v76;
        v76 = 0;
        v78 = v37;
        v74 = v52;
        v77 = v75;
        if ( v52 > 0x40 )
        {
          sub_C43690(&v73, -1, 1);
        }
        else
        {
          v38 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v52;
          if ( !v52 )
            v38 = 0;
          v73 = v38;
        }
        v39 = *a1;
        if ( (unsigned __int8)v39 <= 0x1Cu )
          v40 = *((unsigned __int16 *)a1 + 1);
        else
          v40 = v39 - 29;
        if ( v40 == 25 )
        {
          v80 = v74;
          if ( v74 > 0x40 )
            sub_C43780(&v79, &v73);
          else
            v79 = (unsigned __int64 *)v73;
          sub_C48380(&v79, &v77);
          goto LABEL_41;
        }
        if ( (unsigned int)(v40 - 26) <= 1 )
        {
          v80 = v74;
          if ( v74 > 0x40 )
            sub_C43780(&v79, &v73);
          else
            v79 = (unsigned __int64 *)v73;
          sub_C47AC0(&v79, &v77);
LABEL_41:
          if ( v72 <= 0x40 )
          {
            if ( v71 != v79 )
            {
              result = 0;
LABEL_43:
              if ( v80 > 0x40 && v79 )
              {
                v55 = result;
                j_j___libc_free_0_0(v79);
                result = v55;
              }
              if ( v74 > 0x40 && v73 )
              {
                v56 = result;
                j_j___libc_free_0_0(v73);
                result = v56;
              }
              if ( v78 > 0x40 && v77 )
              {
                v57 = result;
                j_j___libc_free_0_0(v77);
                result = v57;
              }
              if ( v76 > 0x40 && v75 )
              {
                v58 = result;
                j_j___libc_free_0_0(v75);
                result = v58;
              }
              if ( v72 > 0x40 && v71 )
              {
                v59 = result;
                j_j___libc_free_0_0(v71);
                result = v59;
              }
              if ( v70 > 0x40 && v69 )
              {
                v60 = result;
                j_j___libc_free_0_0(v69);
                result = v60;
              }
              if ( v68 > 0x40 && v67 )
              {
                v61 = result;
                j_j___libc_free_0_0(v67);
                result = v61;
              }
              goto LABEL_64;
            }
          }
          else
          {
            result = sub_C43C50(&v71, &v79);
            if ( !(_BYTE)result )
              goto LABEL_43;
          }
          if ( (a1[7] & 0x40) != 0 )
            v47 = (__int64 *)*((_QWORD *)a1 - 1);
          else
            v47 = (__int64 *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
          result = sub_9A6530(*v47, a2, a4, a3);
          goto LABEL_43;
        }
LABEL_136:
        BUG();
      }
      v35 = v75;
    }
    else
    {
      v35 = (unsigned __int64)v65;
    }
    v36 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v34) & ~v35;
    if ( !v34 )
      v36 = 0;
    v75 = v36;
    goto LABEL_99;
  }
LABEL_64:
  if ( v66 > 0x40 )
    goto LABEL_78;
LABEL_65:
  if ( v84 > 0x40 && v83 )
  {
    v62 = result;
    j_j___libc_free_0_0(v83);
    result = v62;
  }
  if ( v82 > 0x40 )
  {
    if ( v81 )
    {
      v63 = result;
      j_j___libc_free_0_0(v81);
      return v63;
    }
  }
  return result;
}
