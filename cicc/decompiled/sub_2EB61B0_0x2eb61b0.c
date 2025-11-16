// Function: sub_2EB61B0
// Address: 0x2eb61b0
//
__int64 __fastcall sub_2EB61B0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 *v10; // rdx
  __int64 *v11; // r12
  __int64 v12; // r13
  __int64 v13; // rdx
  unsigned int v14; // eax
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // r15
  __int64 v19; // rdx
  unsigned int v20; // eax
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  unsigned int v25; // eax
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // r10
  __int64 v31; // rdx
  unsigned int v32; // eax
  __int64 v33; // rax
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r10
  __int64 v37; // r9
  __int64 v38; // r11
  __int64 v39; // rdx
  unsigned int v40; // eax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rcx
  __int64 v44; // r11
  __int64 v45; // r9
  __int64 v46; // r8
  __int64 v47; // r10
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // r8
  __int64 v52; // r9
  __int64 v53; // rcx
  __int64 v54; // r10
  __int64 v55; // r11
  __int64 v56; // rdx
  unsigned int v57; // eax
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // [rsp+0h] [rbp-60h]
  __int64 v66; // [rsp+8h] [rbp-58h]
  __int64 v67; // [rsp+10h] [rbp-50h]
  __int64 v68; // [rsp+10h] [rbp-50h]
  __int64 v69; // [rsp+18h] [rbp-48h]
  __int64 v70; // [rsp+18h] [rbp-48h]
  __int64 v71; // [rsp+18h] [rbp-48h]
  __int64 v72; // [rsp+20h] [rbp-40h]
  __int64 v73; // [rsp+20h] [rbp-40h]
  __int64 v74; // [rsp+20h] [rbp-40h]
  __int64 v75; // [rsp+20h] [rbp-40h]
  __int64 v76; // [rsp+20h] [rbp-40h]
  __int64 v77; // [rsp+20h] [rbp-40h]
  __int64 v78; // [rsp+20h] [rbp-40h]
  __int64 v79; // [rsp+20h] [rbp-40h]
  __int64 *v80; // [rsp+28h] [rbp-38h]

  *(_QWORD *)(sub_2EB5B40((__int64)a1, *(_QWORD *)(*a1 + 8LL), a3, a4, a5, a6) + 16) = a3;
  result = sub_2E6E010(a1, 1);
  v80 = v10;
  if ( (__int64 *)result != v10 )
  {
    v11 = (__int64 *)result;
    while ( 1 )
    {
      while ( 1 )
      {
        v12 = *v11;
        if ( *v11 )
        {
          v13 = (unsigned int)(*(_DWORD *)(v12 + 24) + 1);
          v14 = *(_DWORD *)(v12 + 24) + 1;
        }
        else
        {
          v13 = 0;
          v14 = 0;
        }
        if ( v14 >= *(_DWORD *)(a2 + 56) )
          break;
        result = *(_QWORD *)(a2 + 48);
        if ( !*(_QWORD *)(result + 8 * v13) )
          break;
        if ( v80 == ++v11 )
          return result;
      }
      v18 = *(_QWORD *)(sub_2EB5B40((__int64)a1, *v11, v13, v7, v8, v9) + 16);
      if ( v18 )
      {
        v19 = (unsigned int)(*(_DWORD *)(v18 + 24) + 1);
        v20 = *(_DWORD *)(v18 + 24) + 1;
      }
      else
      {
        v19 = 0;
        v20 = 0;
      }
      if ( v20 < *(_DWORD *)(a2 + 56) )
      {
        v19 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8 * v19);
        if ( v19 )
          goto LABEL_12;
      }
      v23 = *(_QWORD *)(sub_2EB5B40((__int64)a1, v18, v19, v15, v16, v17) + 16);
      if ( v23 )
      {
        v24 = (unsigned int)(*(_DWORD *)(v23 + 24) + 1);
        v25 = *(_DWORD *)(v23 + 24) + 1;
      }
      else
      {
        v24 = 0;
        v25 = 0;
      }
      if ( v25 >= *(_DWORD *)(a2 + 56) )
        break;
      v24 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8 * v24);
      if ( !v24 )
        break;
LABEL_18:
      v19 = sub_2EB4C20(a2, v18, v24);
LABEL_12:
      ++v11;
      result = sub_2EB4C20(a2, v12, v19);
      if ( v80 == v11 )
        return result;
    }
    v72 = v23;
    v26 = sub_2EB5B40((__int64)a1, v23, v24, v21, v22, v23);
    v29 = v72;
    v30 = *(_QWORD *)(v26 + 16);
    if ( v30 )
    {
      v31 = (unsigned int)(*(_DWORD *)(v30 + 24) + 1);
      v32 = *(_DWORD *)(v30 + 24) + 1;
    }
    else
    {
      v31 = 0;
      v32 = 0;
    }
    if ( v32 < *(_DWORD *)(a2 + 56) )
    {
      v31 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8 * v31);
      if ( v31 )
      {
LABEL_24:
        v24 = sub_2EB4C20(a2, v29, v31);
        goto LABEL_18;
      }
    }
    v69 = v72;
    v73 = v30;
    v33 = sub_2EB5B40((__int64)a1, v30, v31, v27, v28, v29);
    v36 = v73;
    v37 = v69;
    v38 = *(_QWORD *)(v33 + 16);
    if ( v38 )
    {
      v39 = (unsigned int)(*(_DWORD *)(v38 + 24) + 1);
      v40 = *(_DWORD *)(v38 + 24) + 1;
    }
    else
    {
      v39 = 0;
      v40 = 0;
    }
    if ( v40 < *(_DWORD *)(a2 + 56) )
    {
      v39 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8 * v39);
      if ( v39 )
      {
LABEL_30:
        v74 = v37;
        v41 = sub_2EB4C20(a2, v36, v39);
        v29 = v74;
        v31 = v41;
        goto LABEL_24;
      }
    }
    v67 = v73;
    v75 = v38;
    v42 = sub_2EB5B40((__int64)a1, v38, v39, v34, v35, v69);
    v44 = v75;
    v45 = v69;
    v46 = *(_QWORD *)(v42 + 16);
    v47 = v67;
    if ( v46 )
    {
      v48 = (unsigned int)(*(_DWORD *)(v46 + 24) + 1);
      if ( (unsigned int)(*(_DWORD *)(v46 + 24) + 1) >= *(_DWORD *)(a2 + 56) )
        goto LABEL_38;
    }
    else
    {
      v48 = 0;
      if ( !*(_DWORD *)(a2 + 56) )
        goto LABEL_38;
    }
    v48 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8 * v48);
    if ( v48 )
    {
LABEL_34:
      v70 = v47;
      v76 = v45;
      v49 = sub_2EB4C20(a2, v44, v48);
      v36 = v70;
      v37 = v76;
      v39 = v49;
      goto LABEL_30;
    }
LABEL_38:
    v66 = v75;
    v77 = *(_QWORD *)(v42 + 16);
    v50 = sub_2EB5B40((__int64)a1, v46, v48, v43, v46, v69);
    v51 = v77;
    v52 = v69;
    v53 = *(_QWORD *)(v50 + 16);
    v54 = v67;
    v55 = v66;
    if ( v53 )
    {
      v56 = (unsigned int)(*(_DWORD *)(v53 + 24) + 1);
      v57 = *(_DWORD *)(v53 + 24) + 1;
    }
    else
    {
      v56 = 0;
      v57 = 0;
    }
    if ( v57 >= *(_DWORD *)(a2 + 56) || (v56 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8 * v56)) == 0 )
    {
      v65 = v77;
      v79 = v53;
      v59 = sub_2EB5B40((__int64)a1, v53, v56, v53, v51, v69);
      v63 = sub_2EB6120((__int64)a1, *(_QWORD *)(v59 + 16), a2, v60, v61, v62);
      v64 = sub_2EB4C20(a2, v79, v63);
      v51 = v65;
      v55 = v66;
      v54 = v67;
      v52 = v69;
      v56 = v64;
    }
    v68 = v55;
    v71 = v54;
    v78 = v52;
    v58 = sub_2EB4C20(a2, v51, v56);
    v44 = v68;
    v47 = v71;
    v45 = v78;
    v48 = v58;
    goto LABEL_34;
  }
  return result;
}
