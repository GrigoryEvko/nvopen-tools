// Function: sub_10FDEB0
// Address: 0x10fdeb0
//
bool __fastcall sub_10FDEB0(unsigned __int8 *a1, unsigned int a2, __int64 a3, __int64 **a4, unsigned __int8 a5)
{
  __int64 v5; // r14
  int v6; // edx
  bool result; // al
  __int64 v8; // r15
  unsigned __int64 v9; // r13
  __int64 **v10; // r12
  unsigned __int8 v11; // bl
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  unsigned __int8 v17; // cl
  unsigned __int64 v18; // r10
  unsigned __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned int v23; // ebx
  _QWORD *v24; // rax
  __int64 **v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  _QWORD *v29; // rax
  unsigned int v30; // ebx
  int v31; // r15d
  unsigned __int8 *v32; // rax
  unsigned __int64 v33; // rdi
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdx
  unsigned __int64 v37; // rax
  _QWORD *v38; // rax
  __int64 v39; // rcx
  unsigned int v40; // edx
  unsigned int v41; // ebx
  _QWORD *v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rdx
  unsigned int v49; // r15d
  __int64 v50; // rax
  __int64 v51; // rdx
  unsigned __int8 **v52; // rax
  __int64 v53; // rdx
  _QWORD *v54; // rax
  unsigned int v55; // r15d
  __int64 v56; // rax
  __int64 v57; // rdx
  unsigned __int64 v58; // [rsp+8h] [rbp-78h]
  unsigned __int8 v59; // [rsp+13h] [rbp-6Dh]
  unsigned __int8 v60; // [rsp+14h] [rbp-6Ch]
  unsigned int v61; // [rsp+14h] [rbp-6Ch]
  unsigned __int64 v62; // [rsp+18h] [rbp-68h]
  unsigned __int8 v63; // [rsp+20h] [rbp-60h]
  unsigned __int64 v64; // [rsp+20h] [rbp-60h]
  __int64 **v65; // [rsp+20h] [rbp-60h]
  int v66; // [rsp+28h] [rbp-58h]
  __int64 v67; // [rsp+30h] [rbp-50h] BYREF
  __int64 v68; // [rsp+38h] [rbp-48h]
  __int64 v69; // [rsp+40h] [rbp-40h] BYREF
  __int64 v70; // [rsp+48h] [rbp-38h]

  while ( 1 )
  {
    v5 = a3;
    v6 = *a1;
    if ( (unsigned int)(v6 - 12) <= 1 )
      return 1;
    v8 = *((_QWORD *)a1 + 1);
    v9 = (unsigned __int64)a1;
    v10 = a4;
    v11 = a5;
    if ( a4 == (__int64 **)v8 )
      break;
    v63 = a5;
    if ( (unsigned __int8)v6 > 0x15u )
    {
      v39 = *((_QWORD *)a1 + 2);
      if ( !v39 )
        return 0;
      result = *(_QWORD *)(v39 + 8) != 0 || (unsigned __int8)v6 <= 0x1Cu;
      if ( result )
        return 0;
      v40 = v6 - 29;
      if ( v40 == 39 )
      {
        v46 = sub_986520((__int64)a1);
        v47 = sub_BCAE30(*(_QWORD *)(*(_QWORD *)v46 + 8LL));
        v68 = v48;
        v67 = v47;
        v49 = sub_CA1930(&v67);
        v50 = sub_BCAE30((__int64)v10);
        v70 = v51;
        v69 = v50;
        if ( v49 % (unsigned __int64)sub_CA1930(&v69) )
          return 0;
        v52 = (unsigned __int8 **)sub_986520((__int64)a1);
        a5 = v11;
        a4 = v10;
        a3 = v5;
        goto LABEL_37;
      }
      if ( v40 > 0x27 )
      {
        if ( v40 != 49 )
          return result;
        v45 = sub_986520((__int64)a1);
        a1 = *(unsigned __int8 **)v45;
        if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v45 + 8LL) + 8LL) - 17 <= 1 )
          return 0;
        a5 = v11;
        a4 = v10;
        a3 = v5;
      }
      else if ( v40 == 25 )
      {
        v53 = *(_QWORD *)(sub_986520((__int64)a1) + 32);
        if ( *(_BYTE *)v53 != 17 )
          return 0;
        v54 = *(_QWORD **)(v53 + 24);
        if ( *(_DWORD *)(v53 + 32) > 0x40u )
          v54 = (_QWORD *)*v54;
        v55 = (_DWORD)v54 + a2;
        v56 = sub_BCAE30((__int64)v10);
        v70 = v57;
        v69 = v56;
        if ( v55 % (unsigned __int64)sub_CA1930(&v69) )
          return 0;
        v52 = (unsigned __int8 **)sub_986520((__int64)a1);
        a5 = v11;
        a4 = v10;
        a3 = v5;
        a2 = v55;
LABEL_37:
        a1 = *v52;
      }
      else
      {
        if ( v40 != 29 )
          return result;
        v41 = a5;
        v42 = (_QWORD *)sub_986520((__int64)a1);
        if ( !(unsigned __int8)sub_10FDEB0(*v42, a2, v5, v10, v41) )
          return 0;
        v43 = sub_986520((__int64)a1);
        a5 = v41;
        a4 = v10;
        a1 = *(unsigned __int8 **)(v43 + 32);
        a3 = v5;
      }
    }
    else
    {
      v12 = sub_BCAE30(v8);
      v68 = v13;
      v67 = v12;
      v62 = (unsigned int)sub_CA1930(&v67);
      v14 = sub_BCAE30((__int64)v10);
      v70 = v15;
      v69 = v14;
      v16 = sub_CA1930(&v69);
      v17 = v63;
      v18 = v16;
      v19 = v62 / v16;
      v66 = v19;
      if ( v19 != 1 )
      {
        v20 = *((_QWORD *)a1 + 1);
        if ( *(_BYTE *)(v20 + 8) != 12 )
        {
          v60 = v63;
          v64 = v18;
          v21 = sub_BCAE30(v20);
          v70 = v22;
          v69 = v21;
          v23 = sub_CA1930(&v69);
          v24 = (_QWORD *)sub_BD5C60(v9);
          v25 = (__int64 **)sub_BCCE00(v24, v23);
          v26 = sub_AD4C90(v9, v25, 0);
          v17 = v60;
          v18 = v64;
          v9 = v26;
        }
        v58 = v18;
        v59 = v17;
        v27 = sub_BCAE30((__int64)v10);
        v70 = v28;
        v69 = v27;
        v61 = sub_CA1930(&v69);
        v29 = (_QWORD *)sub_BD5C60(v9);
        v65 = (__int64 **)sub_BCCE00(v29, v61);
        if ( v58 <= v62 )
        {
          v30 = 0;
          v31 = 0;
          while ( 1 )
          {
            v32 = (unsigned __int8 *)sub_AD64C0(*(_QWORD *)(v9 + 8), v30, 0);
            v33 = sub_AABE40(0x1Au, (unsigned __int8 *)v9, v32);
            if ( !v33 )
              break;
            v34 = sub_AD4C30(v33, v65, 0);
            if ( !(unsigned __int8)sub_10FDEB0(v34, v30 + a2, v5, v10, v59) )
              break;
            ++v31;
            v30 += v61;
            if ( v31 == v66 )
              return 1;
          }
          return 0;
        }
        return 1;
      }
      v44 = sub_AD4C90((unsigned __int64)a1, v10, 0);
      a5 = v11;
      a4 = v10;
      a1 = (unsigned __int8 *)v44;
      a3 = v5;
    }
  }
  if ( (unsigned __int8)v6 <= 0x15u && sub_AC30F0((__int64)a1) )
    return 1;
  v35 = sub_BCAE30(v8);
  v70 = v36;
  v69 = v35;
  v37 = a2 / (unsigned __int64)sub_CA1930(&v69);
  if ( v11 )
    v37 = (unsigned int)(*(_DWORD *)(v5 + 8) + ~(_DWORD)v37);
  v38 = (_QWORD *)(*(_QWORD *)v5 + 8 * v37);
  if ( !*v38 )
  {
    *v38 = a1;
    return 1;
  }
  return 0;
}
