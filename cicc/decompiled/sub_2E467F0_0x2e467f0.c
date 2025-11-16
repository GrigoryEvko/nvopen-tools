// Function: sub_2E467F0
// Address: 0x2e467f0
//
__int64 __fastcall sub_2E467F0(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r15
  unsigned int v4; // r10d
  __int64 *v5; // r11
  __int64 v6; // r13
  __int64 v7; // rdx
  __int64 v8; // rbx
  char v9; // r12
  __int64 v10; // rcx
  int v11; // edx
  unsigned int v12; // esi
  int v13; // eax
  __int64 v14; // rdi
  unsigned int v15; // r8d
  __int64 v16; // rdx
  int v17; // eax
  char v18; // al
  __int64 v19; // r12
  int v20; // edx
  __int64 v21; // rbx
  unsigned int v22; // r14d
  __int64 *v23; // r13
  unsigned int v24; // ecx
  __int64 **v25; // r14
  unsigned int v26; // r8d
  __int64 **v27; // rbx
  unsigned int v28; // esi
  unsigned int v29; // r9d
  unsigned int v30; // r8d
  unsigned int v31; // r15d
  unsigned int v32; // r13d
  __int64 v33; // rdx
  unsigned int v34; // r9d
  char v35; // di
  int v36; // r12d
  __int64 *v37; // rsi
  __int64 v38; // rcx
  __int64 v39; // rcx
  int v40; // r10d
  int v41; // ecx
  __int64 **v42; // r12
  __int64 **v43; // r14
  __int64 *v44; // rsi
  __int64 v45; // rdi
  __int64 v46; // rdi
  int v47; // r9d
  int v48; // edi
  unsigned int v49; // [rsp+4h] [rbp-10Ch]
  __int64 *v50; // [rsp+8h] [rbp-108h]
  __int64 v51; // [rsp+10h] [rbp-100h]
  __int64 *v52; // [rsp+10h] [rbp-100h]
  unsigned int v53; // [rsp+18h] [rbp-F8h]
  char v54; // [rsp+1Ch] [rbp-F4h]
  unsigned int v55; // [rsp+1Ch] [rbp-F4h]
  unsigned int v56; // [rsp+28h] [rbp-E8h]
  __int64 v57; // [rsp+30h] [rbp-E0h]
  __int64 v58; // [rsp+38h] [rbp-D8h]
  __int64 v59; // [rsp+40h] [rbp-D0h]
  __int64 v60; // [rsp+48h] [rbp-C8h]
  unsigned int v61; // [rsp+50h] [rbp-C0h]
  unsigned int v62; // [rsp+50h] [rbp-C0h]
  unsigned int v63; // [rsp+50h] [rbp-C0h]
  unsigned int v64; // [rsp+54h] [rbp-BCh]
  unsigned int v65; // [rsp+58h] [rbp-B8h]
  __int64 *v66; // [rsp+58h] [rbp-B8h]
  unsigned int v67; // [rsp+58h] [rbp-B8h]
  unsigned int v68; // [rsp+58h] [rbp-B8h]
  __int64 *v69; // [rsp+58h] [rbp-B8h]
  __int64 *v70; // [rsp+58h] [rbp-B8h]
  unsigned int v71; // [rsp+58h] [rbp-B8h]
  __int64 v72; // [rsp+58h] [rbp-B8h]
  unsigned int v73; // [rsp+58h] [rbp-B8h]
  __int64 v74; // [rsp+60h] [rbp-B0h]
  __int64 *v75; // [rsp+68h] [rbp-A8h]
  unsigned int v76; // [rsp+68h] [rbp-A8h]
  __int64 *v77; // [rsp+68h] [rbp-A8h]
  unsigned int v78; // [rsp+68h] [rbp-A8h]
  __int64 *v79; // [rsp+68h] [rbp-A8h]
  __int64 *v80; // [rsp+68h] [rbp-A8h]
  unsigned int v81; // [rsp+68h] [rbp-A8h]
  unsigned int v82; // [rsp+68h] [rbp-A8h]
  unsigned int v83; // [rsp+68h] [rbp-A8h]
  __int64 v84; // [rsp+68h] [rbp-A8h]
  unsigned int v85; // [rsp+70h] [rbp-A0h]
  __int64 *v86; // [rsp+70h] [rbp-A0h]
  unsigned int v87; // [rsp+70h] [rbp-A0h]
  __int64 v88; // [rsp+70h] [rbp-A0h]
  __int64 v89; // [rsp+78h] [rbp-98h]
  __int64 v90; // [rsp+80h] [rbp-90h] BYREF
  __int64 v91; // [rsp+88h] [rbp-88h]
  _QWORD v92[4]; // [rsp+A0h] [rbp-70h] BYREF
  _QWORD v93[2]; // [rsp+C0h] [rbp-50h] BYREF
  char v94; // [rsp+D0h] [rbp-40h]

  result = (__int64)(a1 + 22);
  v89 = (__int64)(a1 + 22);
  if ( (*(_DWORD *)(a2 + 40) & 0xFFFFFF) != 0 )
  {
    v3 = 0;
    v4 = *(_DWORD *)(a2 + 40) & 0xFFFFFF;
    v5 = a1;
    v6 = a2;
    do
    {
      v7 = *(_QWORD *)(v6 + 32);
      result = 5 * v3;
      v8 = v7 + 40 * v3;
      if ( *(_BYTE *)v8 )
        goto LABEL_3;
      if ( (*(_WORD *)(v8 + 2) & 0xFF0) != 0 )
        goto LABEL_3;
      v9 = *(_BYTE *)(v8 + 4) & 1;
      if ( v9 )
        goto LABEL_3;
      if ( (*(_BYTE *)(v8 + 3) & 0x30) != 0 )
        goto LABEL_3;
      result = *(unsigned int *)(v8 + 8);
      if ( !(_DWORD)result )
        goto LABEL_3;
      v75 = v5;
      v85 = v4;
      result = sub_2EAB300(v7 + 40 * v3);
      v4 = v85;
      v5 = v75;
      if ( !(_BYTE)result )
        goto LABEL_3;
      v10 = *v75;
      v76 = v85;
      v86 = v5;
      result = sub_2E465E0(v89, v6, *(_DWORD *)(v8 + 8), v10, v5[1], *((_BYTE *)v5 + 24));
      v5 = v86;
      v4 = v76;
      v74 = result;
      if ( !result )
        goto LABEL_3;
      v65 = v76;
      v77 = v86;
      sub_2E44C10((__int64)&v90, result, v86[1], *((_BYTE *)v86 + 24));
      v11 = *(_DWORD *)(v8 + 8);
      v5 = v86;
      v4 = v65;
      v12 = *(_DWORD *)(v90 + 8);
      v60 = v91;
      v87 = *(_DWORD *)(v91 + 8);
      v64 = v87;
      if ( v12 != v11 )
      {
        v13 = sub_E91E30((_QWORD *)*v77, v12, v11);
        result = sub_E91CF0((_QWORD *)*v77, v87, v13);
        v5 = v77;
        v4 = v65;
        v64 = result;
        if ( !(_DWORD)result )
          goto LABEL_3;
      }
      v14 = v5[2];
      if ( (*(_QWORD *)(*(_QWORD *)(v14 + 384) + 8LL * (v87 >> 6)) & (1LL << v87)) != 0 )
      {
        v66 = v5;
        v78 = v4;
        result = sub_2EBF3A0(v14, v87);
        v4 = v78;
        v5 = v66;
        if ( !(_BYTE)result )
          goto LABEL_3;
      }
      v61 = v4;
      v79 = v5;
      sub_2E44C10((__int64)v92, v74, v5[1], *((_BYTE *)v5 + 24));
      v67 = *(_DWORD *)(v92[1] + 8LL);
      result = sub_2E8A250(v6, (unsigned int)v3, v79[1], *v79);
      v5 = v79;
      v15 = v67;
      v4 = v61;
      if ( result )
      {
        if ( v67 - 1 <= 0x3FFFFFFE )
        {
          v16 = *(_QWORD *)result;
          result = v67 >> 3;
          if ( (unsigned int)result < *(unsigned __int16 *)(v16 + 22) )
          {
            result = *(unsigned __int8 *)(*(_QWORD *)(v16 + 8) + result);
            if ( _bittest((const int *)&result, v67 & 7) )
              goto LABEL_19;
          }
        }
        goto LABEL_3;
      }
      v71 = v61;
      v62 = v15;
      result = sub_2E44C10((__int64)v93, v6, v79[1], *((_BYTE *)v79 + 24));
      v5 = v79;
      v4 = v71;
      if ( !v94 )
        goto LABEL_3;
      v24 = *(_DWORD *)(v93[0] + 8LL);
      result = *v79;
      v25 = *(__int64 ***)(*v79 + 280);
      if ( *(__int64 ***)(*v79 + 288) == v25 )
        goto LABEL_3;
      v26 = v62;
      v63 = v71;
      v59 = v8;
      v27 = *(__int64 ***)(*v79 + 288);
      v28 = v26;
      v29 = v26 - 1;
      v58 = v3;
      v30 = v26 & 7;
      v57 = v6;
      v31 = v29;
      v56 = v28 >> 3;
      v32 = v28 >> 3;
      v84 = v28 >> 3;
      v33 = v84;
      result = v24 >> 3;
      v34 = v24 >> 3;
      v72 = (unsigned int)result;
      v35 = v9;
      v36 = *(_DWORD *)(v93[0] + 8LL);
      while ( 1 )
      {
        if ( v31 > 0x3FFFFFFE )
          goto LABEL_42;
        v37 = *v25;
        v38 = **v25;
        result = *(unsigned __int16 *)(v38 + 22);
        if ( v32 >= (unsigned int)result )
          goto LABEL_42;
        v39 = *(_QWORD *)(v38 + 8);
        v40 = *(unsigned __int8 *)(v39 + v33);
        if ( !_bittest(&v40, v30) )
          goto LABEL_42;
        if ( (unsigned int)(v36 - 1) > 0x3FFFFFFE )
          goto LABEL_42;
        if ( (unsigned int)result <= v34 )
          goto LABEL_42;
        result = (int)*(unsigned __int8 *)(v39 + v72) >> (v36 & 7);
        v41 = ((int)*(unsigned __int8 *)(v39 + v72) >> (v36 & 7)) & 1;
        if ( !v41 )
          goto LABEL_42;
        result = *(_QWORD *)(*(_QWORD *)*v5 + 344LL);
        if ( (__int64 (__fastcall *)(__int64, __int64))result != sub_2E44820 )
        {
          v49 = v30;
          v50 = v5;
          v53 = v34;
          v51 = v33;
          v54 = v41;
          result = ((__int64 (*)(void))result)();
          LOBYTE(v41) = v54;
          v33 = v51;
          v34 = v53;
          v5 = v50;
          v30 = v49;
          if ( v37 != (__int64 *)result )
            break;
        }
        v35 = v41;
LABEL_42:
        if ( v27 == ++v25 )
        {
          v4 = v63;
          v8 = v59;
          v3 = v58;
          v6 = v57;
          if ( !v35 )
            goto LABEL_3;
          goto LABEL_19;
        }
      }
      v4 = v63;
      v3 = v58;
      v6 = v57;
      v73 = *(_DWORD *)(v92[0] + 8LL);
      result = *v50;
      v42 = *(__int64 ***)(*v50 + 280);
      if ( *(__int64 ***)(*v50 + 288) == v42 )
        goto LABEL_3;
      v43 = *(__int64 ***)(*v50 + 288);
      while ( 1 )
      {
        v44 = *v42;
        v45 = **v42;
        result = *(unsigned __int16 *)(v45 + 22);
        if ( v56 < (unsigned int)result )
        {
          v46 = *(_QWORD *)(v45 + 8);
          v47 = *(unsigned __int8 *)(v46 + v84);
          if ( _bittest(&v47, v49) )
          {
            if ( v73 - 1 <= 0x3FFFFFFE && (unsigned int)result > v73 >> 3 )
            {
              v48 = *(unsigned __int8 *)(v46 + (v73 >> 3));
              result = v73 & 7;
              if ( _bittest(&v48, result) )
              {
                result = *(_QWORD *)(*(_QWORD *)*v5 + 344LL);
                if ( (__int64 (__fastcall *)(__int64, __int64))result != sub_2E44820 )
                {
                  v52 = v5;
                  v55 = v4;
                  result = ((__int64 (__fastcall *)(__int64, __int64 *, __int64))result)(*v5, v44, v33);
                  v4 = v55;
                  v5 = v52;
                  if ( v44 != (__int64 *)result )
                    break;
                }
              }
            }
          }
        }
        if ( v43 == ++v42 )
        {
          v3 = v58;
          v6 = v57;
          goto LABEL_3;
        }
      }
      v8 = v59;
      v3 = v58;
      v6 = v57;
LABEL_19:
      v68 = v4;
      v80 = v5;
      result = sub_2E45000(v5, v6, v8);
      v5 = v80;
      v4 = v68;
      if ( !(_BYTE)result )
      {
        sub_2E44C10((__int64)v93, v6, v80[1], *((_BYTE *)v80 + 24));
        v5 = v80;
        v4 = v68;
        if ( !v94 )
          goto LABEL_23;
        v17 = sub_2E8E710(v6, v87, *v80, 0, 1);
        v5 = v80;
        v4 = v68;
        if ( v17 == -1
          || (v69 = v80, v81 = v4, result = sub_2E8E710(v6, v87, 0, 0, 0), v4 = v81, v5 = v69, (_DWORD)result != -1) )
        {
LABEL_23:
          v70 = v5;
          v82 = v4;
          sub_2EAB0C0(v8, v64);
          v18 = sub_2EAB300(v60);
          v4 = v82;
          v5 = v70;
          if ( !v18 )
          {
            sub_2EAB350(v8, 0);
            v5 = v70;
            v4 = v82;
          }
          v19 = v74;
          v20 = *(_BYTE *)(v60 + 4) & 1;
          result = v20 | *(_BYTE *)(v8 + 4) & 0xFEu;
          *(_BYTE *)(v8 + 4) = v20 | *(_BYTE *)(v8 + 4) & 0xFE;
          v21 = *(_QWORD *)(v6 + 8);
          if ( v74 != v21 )
          {
            v83 = v4;
            v22 = v87;
            v88 = v6;
            v23 = v5;
            do
            {
              result = sub_2E8D6E0(v19, v22, *v23);
              v19 = *(_QWORD *)(v19 + 8);
            }
            while ( v19 != v21 );
            v5 = v23;
            v4 = v83;
            v6 = v88;
          }
          *((_BYTE *)v5 + 240) = 1;
        }
      }
LABEL_3:
      ++v3;
    }
    while ( v4 > (unsigned int)v3 );
  }
  return result;
}
