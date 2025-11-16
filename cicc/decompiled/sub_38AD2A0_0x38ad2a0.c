// Function: sub_38AD2A0
// Address: 0x38ad2a0
//
__int64 __fastcall sub_38AD2A0(__int64 a1, __int64 *a2, __int64 *a3, double a4, double a5, double a6)
{
  char v7; // r13
  bool v9; // zf
  unsigned __int64 v10; // r14
  __int64 result; // rax
  char v12; // r10
  __int64 v13; // rax
  char v14; // dl
  __int64 v15; // rcx
  __int64 v16; // rdi
  int v17; // r14d
  __int64 v18; // rdx
  __int64 v19; // r15
  __int64 v20; // r14
  __int64 *v21; // r12
  _QWORD *v22; // rax
  char v23; // r10
  __int64 v24; // rbx
  int v25; // r8d
  __int64 v26; // rax
  __int64 *v27; // rax
  __int64 *v28; // rax
  char v29; // r10
  int v30; // r8d
  __int64 *v31; // r11
  __int64 *v32; // rcx
  __int64 *v33; // rax
  __int64 v34; // rdx
  __int64 *v35; // rax
  int v36; // eax
  int v37; // r8d
  int v38; // r9d
  __int64 v39; // rax
  char v40; // dl
  __int64 v41; // rax
  unsigned __int64 v42; // rax
  bool v43; // al
  __int64 v44; // rcx
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 *v47; // rax
  const char *v48; // rax
  int v49; // [rsp+Ch] [rbp-184h]
  char v50; // [rsp+10h] [rbp-180h]
  char v51; // [rsp+18h] [rbp-178h]
  int v52; // [rsp+18h] [rbp-178h]
  char v53; // [rsp+18h] [rbp-178h]
  unsigned __int64 v54; // [rsp+18h] [rbp-178h]
  char v55; // [rsp+18h] [rbp-178h]
  char v56; // [rsp+20h] [rbp-170h]
  unsigned int v57; // [rsp+20h] [rbp-170h]
  __int64 v58; // [rsp+20h] [rbp-170h]
  __int64 v59; // [rsp+20h] [rbp-170h]
  unsigned __int64 v60; // [rsp+30h] [rbp-160h]
  __int64 v61; // [rsp+30h] [rbp-160h]
  unsigned int v62; // [rsp+38h] [rbp-158h]
  unsigned int v63; // [rsp+38h] [rbp-158h]
  char v64; // [rsp+38h] [rbp-158h]
  __int64 v65; // [rsp+48h] [rbp-148h] BYREF
  __int64 *v66; // [rsp+50h] [rbp-140h] BYREF
  __int64 v67; // [rsp+58h] [rbp-138h] BYREF
  _QWORD v68[2]; // [rsp+60h] [rbp-130h] BYREF
  __int16 v69; // [rsp+70h] [rbp-120h]
  const char *v70; // [rsp+80h] [rbp-110h] BYREF
  _BYTE *v71; // [rsp+88h] [rbp-108h]
  _BYTE *v72; // [rsp+90h] [rbp-100h]
  __int64 v73; // [rsp+98h] [rbp-F8h]
  int v74; // [rsp+A0h] [rbp-F0h]
  _BYTE v75[40]; // [rsp+A8h] [rbp-E8h] BYREF
  char *v76; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v77; // [rsp+D8h] [rbp-B8h]
  char v78; // [rsp+E0h] [rbp-B0h] BYREF
  char v79; // [rsp+E1h] [rbp-AFh]

  v7 = 0;
  v9 = *(_DWORD *)(a1 + 64) == 86;
  v65 = 0;
  v66 = 0;
  if ( v9 )
  {
    v7 = 1;
    *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  }
  v10 = *(_QWORD *)(a1 + 56);
  v67 = 0;
  v79 = 1;
  v76 = "expected type";
  v78 = 3;
  if ( (unsigned __int8)sub_3891B00(a1, &v67, (__int64)&v76, 0) )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 4, "expected comma after getelementptr's type") )
    return 1;
  v60 = *(_QWORD *)(a1 + 56);
  v12 = sub_38AB270((__int64 **)a1, &v65, a3, a4, a5, a6);
  if ( v12 )
    return 1;
  v13 = *(_QWORD *)v65;
  v14 = *(_BYTE *)(*(_QWORD *)v65 + 8LL);
  v15 = *(_QWORD *)v65;
  if ( v14 == 16 )
  {
    v15 = **(_QWORD **)(v13 + 16);
    v14 = *(_BYTE *)(v15 + 8);
  }
  if ( v14 == 15 )
  {
    v16 = v67;
    if ( v67 == *(_QWORD *)(v15 + 24) )
    {
      v17 = 0;
      v76 = &v78;
      v77 = 0x1000000000LL;
      if ( *(_BYTE *)(v13 + 8) == 16 )
        v17 = *(_DWORD *)(v13 + 32);
      if ( *(_DWORD *)(a1 + 64) == 4 )
      {
        v59 = a1 + 8;
        while ( 1 )
        {
          v36 = sub_3887100(v59);
          *(_DWORD *)(a1 + 64) = v36;
          if ( v36 == 376 )
            break;
          v54 = *(_QWORD *)(a1 + 56);
          v12 = sub_38AB270((__int64 **)a1, &v66, a3, a4, a5, a6);
          if ( v12 )
          {
            result = 1;
            goto LABEL_35;
          }
          v39 = *v66;
          v40 = *(_BYTE *)(*v66 + 8);
          if ( v40 == 16 )
          {
            if ( *(_BYTE *)(**(_QWORD **)(v39 + 16) + 8LL) != 11 )
            {
LABEL_63:
              BYTE1(v72) = 1;
              v48 = "getelementptr index must be an integer";
LABEL_64:
              v70 = v48;
              LOBYTE(v72) = 3;
              result = (unsigned __int8)sub_38814C0(v59, v54, (__int64)&v70);
              goto LABEL_35;
            }
            v46 = *(_QWORD *)(v39 + 32);
            if ( v17 && v17 != (_DWORD)v46 )
            {
              BYTE1(v72) = 1;
              v48 = "getelementptr vector index has a wrong number of elements";
              goto LABEL_64;
            }
            v17 = v46;
          }
          else if ( v40 != 11 )
          {
            goto LABEL_63;
          }
          v41 = (unsigned int)v77;
          if ( (unsigned int)v77 >= HIDWORD(v77) )
          {
            sub_16CD150((__int64)&v76, &v78, 0, 8, v37, v38);
            v41 = (unsigned int)v77;
            v12 = 0;
          }
          *(_QWORD *)&v76[8 * v41] = v66;
          v9 = *(_DWORD *)(a1 + 64) == 4;
          v18 = (unsigned int)(v77 + 1);
          LODWORD(v77) = v77 + 1;
          if ( !v9 )
          {
            v16 = v67;
            goto LABEL_46;
          }
        }
        v16 = v67;
        v18 = (unsigned int)v77;
        v12 = 1;
LABEL_46:
        v70 = 0;
        v71 = v75;
        v72 = v75;
        v73 = 4;
        v74 = 0;
        if ( (_DWORD)v18 )
        {
          v42 = *(unsigned __int8 *)(v16 + 8);
          if ( (unsigned __int8)v42 > 0xFu || (v44 = 35454, !_bittest64(&v44, v42)) )
          {
            if ( (unsigned int)(v42 - 13) > 1 && (_DWORD)v42 != 16
              || (v55 = v12, v43 = sub_16435F0(v16, (__int64)&v70), v12 = v55, !v43) )
            {
              v68[0] = "base element of getelementptr must be sized";
              v69 = 259;
              result = (unsigned __int8)sub_38814C0(v59, v60, (__int64)v68);
              goto LABEL_33;
            }
            v16 = v67;
            v18 = (unsigned int)v77;
          }
        }
      }
      else
      {
        v18 = 0;
        v70 = 0;
        v71 = v75;
        v72 = v75;
        v73 = 4;
        v74 = 0;
      }
      v56 = v12;
      if ( sub_15F9F50(v16, (__int64)v76, v18) )
      {
        v19 = (unsigned int)v77;
        v20 = v67;
        v69 = 257;
        v21 = (__int64 *)v76;
        v61 = v65;
        if ( !v67 )
        {
          v45 = *(_QWORD *)v65;
          if ( *(_BYTE *)(*(_QWORD *)v65 + 8LL) == 16 )
            v45 = **(_QWORD **)(v45 + 16);
          v20 = *(_QWORD *)(v45 + 24);
        }
        v51 = v56;
        v57 = v77 + 1;
        v22 = sub_1648A60(72, (int)v77 + 1);
        v23 = v51;
        v24 = (__int64)v22;
        if ( v22 )
        {
          v25 = v57;
          v58 = (__int64)&v22[-3 * v57];
          v26 = *(_QWORD *)v61;
          if ( *(_BYTE *)(*(_QWORD *)v61 + 8LL) == 16 )
            v26 = **(_QWORD **)(v26 + 16);
          v49 = v25;
          v50 = v51;
          v52 = *(_DWORD *)(v26 + 8) >> 8;
          v27 = (__int64 *)sub_15F9F50(v20, (__int64)v21, v19);
          v28 = (__int64 *)sub_1646BA0(v27, v52);
          v29 = v50;
          v30 = v49;
          v31 = v28;
          if ( *(_BYTE *)(*(_QWORD *)v61 + 8LL) == 16 )
          {
            v47 = sub_16463B0(v28, *(_QWORD *)(*(_QWORD *)v61 + 32LL));
            v29 = v50;
            v30 = v49;
            v31 = v47;
          }
          else
          {
            v32 = &v21[v19];
            if ( v21 != v32 )
            {
              v33 = v21;
              while ( 1 )
              {
                v34 = *(_QWORD *)*v33;
                if ( *(_BYTE *)(v34 + 8) == 16 )
                  break;
                if ( v32 == ++v33 )
                  goto LABEL_29;
              }
              v35 = sub_16463B0(v31, *(_QWORD *)(v34 + 32));
              v30 = v49;
              v29 = v50;
              v31 = v35;
            }
          }
LABEL_29:
          v53 = v29;
          sub_15F1EA0(v24, (__int64)v31, 32, v58, v30, 0);
          *(_QWORD *)(v24 + 56) = v20;
          *(_QWORD *)(v24 + 64) = sub_15F9F50(v20, (__int64)v21, v19);
          sub_15F9CE0(v24, v61, v21, v19, (__int64)v68);
          v23 = v53;
        }
        *a2 = v24;
        if ( v7 )
        {
          v64 = v23;
          sub_15FA2E0(v24, 1);
          v23 = v64;
        }
        result = 2 * (unsigned int)(v23 != 0);
      }
      else
      {
        v68[0] = "invalid getelementptr indices";
        v69 = 259;
        result = (unsigned __int8)sub_38814C0(a1 + 8, v60, (__int64)v68);
      }
LABEL_33:
      if ( v72 != v71 )
      {
        v62 = result;
        _libc_free((unsigned __int64)v72);
        result = v62;
      }
LABEL_35:
      if ( v76 != &v78 )
      {
        v63 = result;
        _libc_free((unsigned __int64)v76);
        return v63;
      }
    }
    else
    {
      v79 = 1;
      v76 = "explicit pointee type doesn't match operand's pointee type";
      v78 = 3;
      return (unsigned __int8)sub_38814C0(a1 + 8, v10, (__int64)&v76);
    }
  }
  else
  {
    v79 = 1;
    v76 = "base of getelementptr must be a pointer";
    v78 = 3;
    return (unsigned __int8)sub_38814C0(a1 + 8, v60, (__int64)&v76);
  }
  return result;
}
