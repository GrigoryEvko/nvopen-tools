// Function: sub_1CD5660
// Address: 0x1cd5660
//
__int64 __fastcall sub_1CD5660(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 a15)
{
  __int64 result; // rax
  __int64 v16; // rdx
  __int64 v18; // r13
  __int64 v19; // r12
  _QWORD *v20; // rsi
  _QWORD *v21; // rax
  _QWORD *v22; // r8
  __int64 v23; // rcx
  unsigned int v24; // edi
  _QWORD *v25; // rdx
  _QWORD *v26; // r10
  __int64 v27; // rbx
  __int64 *v28; // rax
  __int64 v29; // rbx
  __int64 *v30; // rax
  __int64 v31; // rbx
  __int64 *v32; // rax
  unsigned __int64 v33; // rbx
  __int64 v34; // rbx
  __int64 *v35; // rax
  __int64 v36; // rbx
  __int64 v37; // r15
  __int64 *v38; // r15
  __int64 *v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // r8
  char v43; // al
  bool v44; // zf
  __int64 v45; // rax
  __int64 *v46; // rax
  double v47; // xmm4_8
  double v48; // xmm5_8
  _QWORD *v49; // rdi
  __int64 v50; // r15
  __int64 *v51; // rax
  __int64 v52; // rdi
  __int64 **v53; // rsi
  __int64 *v54; // rax
  _QWORD *v55; // r15
  __int64 *v56; // rax
  __int64 *v57; // rax
  __int64 v58; // rdi
  __int64 **v59; // rsi
  int v60; // edx
  int v61; // r9d
  __int64 v62; // [rsp+0h] [rbp-110h]
  __int64 *v66; // [rsp+28h] [rbp-E8h]
  __int64 *v67; // [rsp+28h] [rbp-E8h]
  __int64 v68; // [rsp+28h] [rbp-E8h]
  __int64 v69; // [rsp+28h] [rbp-E8h]
  _QWORD *v70; // [rsp+28h] [rbp-E8h]
  __int64 v71; // [rsp+28h] [rbp-E8h]
  __int64 v72; // [rsp+30h] [rbp-E0h]
  int v73; // [rsp+30h] [rbp-E0h]
  __int64 v74; // [rsp+30h] [rbp-E0h]
  __int64 v75; // [rsp+38h] [rbp-D8h]
  __int64 *v76; // [rsp+38h] [rbp-D8h]
  __int64 v78; // [rsp+50h] [rbp-C0h]
  __int64 v80; // [rsp+68h] [rbp-A8h] BYREF
  _QWORD *v81; // [rsp+70h] [rbp-A0h] BYREF
  _QWORD *v82; // [rsp+78h] [rbp-98h] BYREF
  __int64 *v83[4]; // [rsp+80h] [rbp-90h] BYREF
  __int64 *v84[4]; // [rsp+A0h] [rbp-70h] BYREF
  __int64 *v85[2]; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v86; // [rsp+D0h] [rbp-40h]

  result = *a4;
  v16 = (a4[1] - *a4) >> 3;
  if ( (_DWORD)v16 )
  {
    v18 = 0;
    v78 = 8LL * (unsigned int)(v16 - 1);
    while ( 1 )
    {
      v19 = *(_QWORD *)(result + v18);
      result = *(unsigned __int8 *)(v19 + 16);
      v80 = v19;
      if ( (unsigned __int8)result <= 0x17u )
        goto LABEL_23;
      result = (unsigned int)(result - 35);
      if ( (result & 0xFD) != 0 )
        goto LABEL_23;
      if ( (*(_BYTE *)(v19 + 23) & 0x40) != 0 )
      {
        v20 = **(_QWORD ***)(v19 - 8);
        v81 = v20;
        v21 = *(_QWORD **)(v19 - 8);
      }
      else
      {
        v21 = (_QWORD *)(v19 - 24LL * (*(_DWORD *)(v19 + 20) & 0xFFFFFFF));
        v20 = (_QWORD *)*v21;
        v81 = (_QWORD *)*v21;
      }
      v22 = (_QWORD *)v21[3];
      result = *(unsigned int *)(a5 + 24);
      v82 = v22;
      if ( !(_DWORD)result )
        goto LABEL_23;
      v23 = *(_QWORD *)(a5 + 8);
      v24 = (result - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v25 = (_QWORD *)(v23 + 16LL * v24);
      v26 = (_QWORD *)*v25;
      if ( v20 != (_QWORD *)*v25 )
        break;
LABEL_9:
      result *= 16;
      v27 = v23 + result;
      if ( v25 == (_QWORD *)(v23 + result) )
        goto LABEL_23;
      result = (__int64)sub_1CD1E70(v85, (__int64 *)a5, (__int64)v22);
      if ( v27 == v86 )
        goto LABEL_23;
      v28 = sub_1CD52D0(a5, (__int64 *)&v81);
      v29 = *(_QWORD *)(v28[1] + 8) - *(_QWORD *)v28[1];
      v30 = sub_1CD52D0(a5, (__int64 *)&v82);
      v31 = (v29 >> 3) + ((__int64)(*(_QWORD *)(v30[1] + 8) - *(_QWORD *)v30[1]) >> 3);
      v32 = sub_1CD52D0(a5, &v80);
      v33 = ((__int64)(*(_QWORD *)(v32[1] + 8) - *(_QWORD *)v32[1]) >> 3) + v31;
      v66 = (__int64 *)(*(_QWORD *)(a6 + 8) + 16LL * *(unsigned int *)(a6 + 24));
      sub_1CD1F20(v83, (__int64 *)a6, (__int64)v81);
      if ( v83[2] != v66 )
        v33 -= (sub_1CD5530(a6, (__int64 *)&v81)[1] == 0) - 1LL;
      v67 = (__int64 *)(*(_QWORD *)(a6 + 8) + 16LL * *(unsigned int *)(a6 + 24));
      sub_1CD1F20(v84, (__int64 *)a6, (__int64)v82);
      if ( v84[2] != v67 )
        v33 -= (sub_1CD5530(a6, (__int64 *)&v82)[1] == 0) - 1LL;
      v68 = *(_QWORD *)(a6 + 8) + 16LL * *(unsigned int *)(a6 + 24);
      sub_1CD1F20(v85, (__int64 *)a6, v80);
      result = v68;
      if ( v86 != v68 )
      {
        result = (__int64)sub_1CD5530(a6, &v80);
        v33 -= (*(_QWORD *)(result + 8) == 0) - 1LL;
      }
      if ( v33 <= 4 )
        goto LABEL_23;
      v34 = *(_QWORD *)(a6 + 8) + 16LL * *(unsigned int *)(a6 + 24);
      sub_1CD1F20(v85, (__int64 *)a6, v19);
      if ( v86 != v34
        && (v84[0] = (__int64 *)v19, sub_1CD5400(a6, (__int64 *)v84)[1])
        && (v85[0] = (__int64 *)v19, v35 = sub_1CD5400(a6, (__int64 *)v85), v36 = v35[1], *(_BYTE *)(v36 + 16) == 77) )
      {
        result = *(_QWORD *)(sub_1455EB0(v35[1], a3) + 8);
        if ( !result || *(_QWORD *)(result + 8) )
          goto LABEL_23;
      }
      else
      {
        v36 = 0;
      }
      v37 = *(_QWORD *)(a6 + 8) + 16LL * *(unsigned int *)(a6 + 24);
      sub_1CD1F20(v85, (__int64 *)a6, (__int64)v81);
      if ( v86 == v37 || (v38 = (__int64 *)sub_1CD5530(a6, (__int64 *)&v81)[1]) == 0 )
      {
        v55 = v81;
        v71 = *v81;
        v56 = sub_1CD52D0(a5, (__int64 *)&v81);
        v85[0] = (__int64 *)sub_1CD0910(**(_QWORD **)v56[1], v71, (__int64)v55, a2, a3);
        v57 = sub_1CD5530(a6, (__int64 *)&v81);
        v57[1] = (__int64)v85[0];
        v58 = sub_1CD52D0(a5, (__int64 *)&v81)[1];
        v59 = *(__int64 ***)(v58 + 8);
        if ( v59 == *(__int64 ***)(v58 + 16) )
        {
          sub_1769D70(v58, v59, v85);
          v38 = v85[0];
        }
        else
        {
          v38 = v85[0];
          if ( v59 )
          {
            *v59 = v85[0];
            v59 = *(__int64 ***)(v58 + 8);
          }
          *(_QWORD *)(v58 + 8) = v59 + 1;
        }
      }
      v72 = *(_QWORD *)(a6 + 8) + 16LL * *(unsigned int *)(a6 + 24);
      sub_1CD1F20(v85, (__int64 *)a6, (__int64)v82);
      if ( v86 == v72 || (v39 = (__int64 *)sub_1CD5530(a6, (__int64 *)&v82)[1]) == 0 )
      {
        v70 = v82;
        v74 = *v82;
        v51 = sub_1CD52D0(a5, (__int64 *)&v82);
        v85[0] = (__int64 *)sub_1CD0910(**(_QWORD **)v51[1], v74, (__int64)v70, a2, a3);
        v52 = sub_1CD52D0(a5, (__int64 *)&v82)[1];
        v53 = *(__int64 ***)(v52 + 8);
        if ( v53 == *(__int64 ***)(v52 + 16) )
        {
          sub_1769D70(v52, v53, v85);
        }
        else
        {
          if ( v53 )
          {
            *v53 = v85[0];
            v53 = *(__int64 ***)(v52 + 8);
          }
          *(_QWORD *)(v52 + 8) = v53 + 1;
        }
        v54 = sub_1CD5530(a6, (__int64 *)&v82);
        v39 = v85[0];
        v54[1] = (__int64)v85[0];
      }
      v75 = (__int64)v39;
      v40 = sub_157ED20(a1);
      v41 = v75;
      v42 = v40;
      if ( *((_BYTE *)v38 + 16) != 77 || *(_BYTE *)(v75 + 16) != 77 )
      {
        v43 = sub_15CCEE0(a15, (__int64)v38, v75);
        v41 = v75;
        v44 = v43 == 0;
        v45 = (__int64)v38;
        if ( !v44 )
          v45 = v75;
        v42 = *(_QWORD *)(v45 + 32);
        if ( v42 )
          v42 -= 24;
      }
      v85[0] = (__int64 *)"baseIV";
      LOWORD(v86) = 259;
      v62 = v42;
      v69 = v41;
      v73 = 2 * (*(_BYTE *)(v19 + 16) != 35) + 11;
      v84[0] = (__int64 *)v19;
      v76 = sub_1CD5400(a6, (__int64 *)v84);
      result = sub_15FB440(v73, v38, v69, (__int64)v85, v62);
      v76[1] = result;
      if ( !v36 )
        goto LABEL_23;
      v85[0] = (__int64 *)v19;
      v46 = sub_1CD5400(a6, (__int64 *)v85);
      sub_164D160(v36, v46[1], a7, a8, a9, a10, v47, v48, a13, a14);
      v49 = (_QWORD *)v36;
      v50 = sub_1455EB0(v36, a3);
      if ( *(_BYTE *)(v50 + 16) > 0x17u )
      {
        sub_15F20C0((_QWORD *)v36);
        v49 = (_QWORD *)v50;
      }
      result = sub_15F20C0(v49);
      if ( v18 == v78 )
        return result;
LABEL_24:
      v18 += 8;
      result = *a4;
    }
    v60 = 1;
    while ( v26 != (_QWORD *)-8LL )
    {
      v61 = v60 + 1;
      v24 = (result - 1) & (v60 + v24);
      v25 = (_QWORD *)(v23 + 16LL * v24);
      v26 = (_QWORD *)*v25;
      if ( v20 == (_QWORD *)*v25 )
        goto LABEL_9;
      v60 = v61;
    }
LABEL_23:
    if ( v18 == v78 )
      return result;
    goto LABEL_24;
  }
  return result;
}
