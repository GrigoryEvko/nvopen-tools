// Function: sub_B1E720
// Address: 0xb1e720
//
__int64 __fastcall sub_B1E720(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 *v4; // rdx
  __int64 *v5; // r12
  __int64 v6; // r13
  __int64 v7; // rdx
  unsigned int v8; // eax
  __int64 v9; // r15
  __int64 v10; // rdx
  unsigned int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // r9
  __int64 v14; // rdx
  unsigned int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r9
  __int64 v19; // r10
  __int64 v20; // rdx
  unsigned int v21; // eax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // r10
  __int64 v25; // r9
  __int64 v26; // r11
  __int64 v27; // rdx
  unsigned int v28; // eax
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // r11
  __int64 v33; // r9
  __int64 v34; // r8
  __int64 v35; // r10
  __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rcx
  __int64 v43; // r10
  __int64 v44; // r11
  __int64 v45; // rdx
  unsigned int v46; // eax
  __int64 v47; // rdx
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // [rsp+0h] [rbp-60h]
  __int64 v53; // [rsp+8h] [rbp-58h]
  __int64 v54; // [rsp+10h] [rbp-50h]
  __int64 v55; // [rsp+10h] [rbp-50h]
  __int64 v56; // [rsp+18h] [rbp-48h]
  __int64 v57; // [rsp+18h] [rbp-48h]
  __int64 v58; // [rsp+18h] [rbp-48h]
  __int64 v59; // [rsp+20h] [rbp-40h]
  __int64 v60; // [rsp+20h] [rbp-40h]
  __int64 v61; // [rsp+20h] [rbp-40h]
  __int64 v62; // [rsp+20h] [rbp-40h]
  __int64 v63; // [rsp+20h] [rbp-40h]
  __int64 v64; // [rsp+20h] [rbp-40h]
  __int64 v65; // [rsp+20h] [rbp-40h]
  __int64 v66; // [rsp+20h] [rbp-40h]
  __int64 *v67; // [rsp+28h] [rbp-38h]

  *(_QWORD *)(sub_B1E0B0((__int64)a1, *(_QWORD *)(*a1 + 8LL)) + 16) = a3;
  result = sub_B1B2D0(a1, 1);
  v67 = v4;
  if ( (__int64 *)result != v4 )
  {
    v5 = (__int64 *)result;
    while ( 1 )
    {
      while ( 1 )
      {
        v6 = *v5;
        if ( *v5 )
        {
          v7 = (unsigned int)(*(_DWORD *)(v6 + 44) + 1);
          v8 = *(_DWORD *)(v6 + 44) + 1;
        }
        else
        {
          v7 = 0;
          v8 = 0;
        }
        if ( v8 >= *(_DWORD *)(a2 + 32) )
          break;
        result = *(_QWORD *)(a2 + 24);
        if ( !*(_QWORD *)(result + 8 * v7) )
          break;
        if ( v67 == ++v5 )
          return result;
      }
      v9 = *(_QWORD *)(sub_B1E0B0((__int64)a1, *v5) + 16);
      if ( v9 )
      {
        v10 = (unsigned int)(*(_DWORD *)(v9 + 44) + 1);
        v11 = *(_DWORD *)(v9 + 44) + 1;
      }
      else
      {
        v10 = 0;
        v11 = 0;
      }
      if ( v11 < *(_DWORD *)(a2 + 32) )
      {
        v12 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 8 * v10);
        if ( v12 )
          goto LABEL_12;
      }
      v13 = *(_QWORD *)(sub_B1E0B0((__int64)a1, v9) + 16);
      if ( v13 )
      {
        v14 = (unsigned int)(*(_DWORD *)(v13 + 44) + 1);
        v15 = *(_DWORD *)(v13 + 44) + 1;
      }
      else
      {
        v14 = 0;
        v15 = 0;
      }
      if ( v15 >= *(_DWORD *)(a2 + 32) )
        break;
      v16 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 8 * v14);
      if ( !v16 )
        break;
LABEL_18:
      v12 = sub_B1B5D0(a2, v9, v16);
LABEL_12:
      ++v5;
      result = sub_B1B5D0(a2, v6, v12);
      if ( v67 == v5 )
        return result;
    }
    v59 = v13;
    v17 = sub_B1E0B0((__int64)a1, v13);
    v18 = v59;
    v19 = *(_QWORD *)(v17 + 16);
    if ( v19 )
    {
      v20 = (unsigned int)(*(_DWORD *)(v19 + 44) + 1);
      v21 = *(_DWORD *)(v19 + 44) + 1;
    }
    else
    {
      v20 = 0;
      v21 = 0;
    }
    if ( v21 < *(_DWORD *)(a2 + 32) )
    {
      v22 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 8 * v20);
      if ( v22 )
      {
LABEL_24:
        v16 = sub_B1B5D0(a2, v18, v22);
        goto LABEL_18;
      }
    }
    v56 = v59;
    v60 = v19;
    v23 = sub_B1E0B0((__int64)a1, v19);
    v24 = v60;
    v25 = v56;
    v26 = *(_QWORD *)(v23 + 16);
    if ( v26 )
    {
      v27 = (unsigned int)(*(_DWORD *)(v26 + 44) + 1);
      v28 = *(_DWORD *)(v26 + 44) + 1;
    }
    else
    {
      v27 = 0;
      v28 = 0;
    }
    if ( v28 < *(_DWORD *)(a2 + 32) )
    {
      v29 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 8 * v27);
      if ( v29 )
      {
LABEL_30:
        v61 = v25;
        v30 = sub_B1B5D0(a2, v24, v29);
        v18 = v61;
        v22 = v30;
        goto LABEL_24;
      }
    }
    v54 = v60;
    v62 = v26;
    v31 = sub_B1E0B0((__int64)a1, v26);
    v32 = v62;
    v33 = v56;
    v34 = *(_QWORD *)(v31 + 16);
    v35 = v54;
    if ( v34 )
    {
      v36 = (unsigned int)(*(_DWORD *)(v34 + 44) + 1);
      if ( (unsigned int)(*(_DWORD *)(v34 + 44) + 1) >= *(_DWORD *)(a2 + 32) )
        goto LABEL_38;
    }
    else
    {
      v36 = 0;
      if ( !*(_DWORD *)(a2 + 32) )
        goto LABEL_38;
    }
    v37 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 8 * v36);
    if ( v37 )
    {
LABEL_34:
      v57 = v35;
      v63 = v33;
      v38 = sub_B1B5D0(a2, v32, v37);
      v24 = v57;
      v25 = v63;
      v29 = v38;
      goto LABEL_30;
    }
LABEL_38:
    v53 = v62;
    v64 = *(_QWORD *)(v31 + 16);
    v39 = sub_B1E0B0((__int64)a1, v34);
    v40 = v64;
    v41 = v56;
    v42 = *(_QWORD *)(v39 + 16);
    v43 = v54;
    v44 = v53;
    if ( v42 )
    {
      v45 = (unsigned int)(*(_DWORD *)(v42 + 44) + 1);
      v46 = *(_DWORD *)(v42 + 44) + 1;
    }
    else
    {
      v45 = 0;
      v46 = 0;
    }
    if ( v46 >= *(_DWORD *)(a2 + 32) || (v47 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 8 * v45)) == 0 )
    {
      v52 = v64;
      v66 = v42;
      v49 = sub_B1E0B0((__int64)a1, v42);
      v50 = sub_B1E690((__int64)a1, *(_QWORD *)(v49 + 16), a2);
      v51 = sub_B1B5D0(a2, v66, v50);
      v40 = v52;
      v44 = v53;
      v43 = v54;
      v41 = v56;
      v47 = v51;
    }
    v55 = v44;
    v58 = v43;
    v65 = v41;
    v48 = sub_B1B5D0(a2, v40, v47);
    v32 = v55;
    v35 = v58;
    v33 = v65;
    v37 = v48;
    goto LABEL_34;
  }
  return result;
}
