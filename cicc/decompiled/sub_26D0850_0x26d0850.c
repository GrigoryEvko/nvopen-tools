// Function: sub_26D0850
// Address: 0x26d0850
//
void __fastcall sub_26D0850(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 i; // rbx
  __int64 *v8; // rax
  int *v9; // rsi
  __int64 v10; // rdx
  __int64 *v11; // rax
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // r12
  int *v15; // rsi
  __int64 v16; // rdx
  __int64 *v17; // rax
  __int64 v18; // rdi
  _QWORD *v19; // r13
  unsigned __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rax
  unsigned int v24; // ebx
  _QWORD *v25; // r15
  __int64 v26; // r12
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r13
  unsigned int v30; // eax
  __int64 v31; // r14
  __int64 v32; // r13
  _QWORD *v33; // rdi
  __int64 v34; // rax
  unsigned int *v35; // rsi
  __int64 v36; // r14
  unsigned __int64 v37; // rcx
  _QWORD *v38; // r8
  __int64 *v39; // rax
  __int64 v40; // rax
  unsigned __int64 v41; // rax
  __int64 v42; // rax
  size_t v43; // rdx
  size_t v44; // r14
  int *v45; // rsi
  size_t v46; // rdx
  __int64 v47; // rdx
  _QWORD *v48; // rax
  __int64 v49; // r12
  _QWORD *v50; // rdx
  _QWORD *v51; // rcx
  _QWORD *v52; // r14
  _QWORD *v53; // r13
  __int64 v54; // rax
  __int64 v55; // rdx
  _QWORD *v56; // r12
  unsigned __int64 v57; // rbx
  int *v58; // r15
  __int64 v59; // rax
  int *v60; // rax
  __int64 v61; // [rsp+0h] [rbp-1B0h]
  __int64 v62; // [rsp+8h] [rbp-1A8h]
  _QWORD *v63; // [rsp+10h] [rbp-1A0h]
  size_t v64; // [rsp+10h] [rbp-1A0h]
  char v65; // [rsp+28h] [rbp-188h]
  __int64 v67; // [rsp+40h] [rbp-170h]
  __int64 v68; // [rsp+48h] [rbp-168h]
  __int64 v69; // [rsp+50h] [rbp-160h]
  __int64 v70; // [rsp+50h] [rbp-160h]
  _QWORD *v71; // [rsp+58h] [rbp-158h]
  int *v72; // [rsp+58h] [rbp-158h]
  int *v73; // [rsp+58h] [rbp-158h]
  _QWORD *v74; // [rsp+58h] [rbp-158h]
  __int64 v75; // [rsp+60h] [rbp-150h] BYREF
  __int64 v76; // [rsp+68h] [rbp-148h] BYREF
  _QWORD v77[2]; // [rsp+70h] [rbp-140h] BYREF
  unsigned __int64 v78; // [rsp+80h] [rbp-130h] BYREF
  __int64 v79[2]; // [rsp+90h] [rbp-120h] BYREF
  unsigned __int64 v80; // [rsp+A0h] [rbp-110h]
  unsigned __int64 v81; // [rsp+A8h] [rbp-108h]
  __int64 v82; // [rsp+B0h] [rbp-100h]
  unsigned __int64 *v83; // [rsp+B8h] [rbp-F8h]
  __int64 *v84; // [rsp+C0h] [rbp-F0h]
  __int64 v85; // [rsp+C8h] [rbp-E8h]
  __int64 v86; // [rsp+D0h] [rbp-E0h]
  __int64 v87; // [rsp+D8h] [rbp-D8h]
  int v88[52]; // [rsp+E0h] [rbp-D0h] BYREF

  v3 = a1;
  *(_QWORD *)(a1 + 40) = a1 + 24;
  *(_QWORD *)(a1 + 48) = a1 + 24;
  *(_QWORD *)(a1 + 72) = a1 + 64;
  *(_QWORD *)(a1 + 64) = a1 + 64;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_DWORD *)(a1 + 112) = 0;
  v79[0] = 0;
  v79[1] = 0;
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  sub_26C4970(v79, 0);
  v4 = sub_317E690(a2);
  v5 = sub_317E450(v4);
  v6 = *(_QWORD *)(v5 + 24);
  for ( i = v5 + 8; v6 != i; v6 = sub_220EEE0(v6) )
  {
    while ( 1 )
    {
      *(_QWORD *)v88 = v6 + 40;
      v9 = (int *)((__int64 (*)(void))sub_317E460)();
      sub_26D04C0(a1, v9, v10);
      v8 = v84;
      if ( v84 != (__int64 *)(v86 - 8) )
        break;
      sub_26C4A60((unsigned __int64 *)v79, v88);
      v6 = sub_220EEE0(v6);
      if ( v6 == i )
        goto LABEL_8;
    }
    if ( v84 )
    {
      *v84 = *(_QWORD *)v88;
      v8 = v84;
    }
    v84 = v8 + 1;
  }
LABEL_8:
  v11 = (__int64 *)v80;
  v12 = a1;
  if ( v84 != (__int64 *)v80 )
  {
    while ( 1 )
    {
      v69 = *v11;
      if ( v11 == (__int64 *)(v82 - 8) )
      {
        j_j___libc_free_0(v81);
        v47 = *++v83 + 512;
        v81 = *v83;
        v82 = v47;
        v80 = v81;
      }
      else
      {
        v80 = (unsigned __int64)(v11 + 1);
      }
      v67 = sub_317E470(v69);
      v13 = sub_317E450(v69);
      v14 = *(_QWORD *)(v13 + 24);
      v68 = v13 + 8;
      if ( v13 + 8 != v14 )
        break;
LABEL_50:
      v11 = (__int64 *)v80;
      if ( v84 == (__int64 *)v80 )
      {
        v3 = v12;
        goto LABEL_52;
      }
    }
    while ( 1 )
    {
      v75 = v14 + 40;
      v15 = (int *)((__int64 (*)(void))sub_317E460)();
      sub_26D04C0(v12, v15, v16);
      v17 = v84;
      if ( v84 == (__int64 *)(v86 - 8) )
      {
        sub_26C4A60((unsigned __int64 *)v79, &v75);
        v18 = v75;
      }
      else
      {
        v18 = v75;
        if ( v84 )
        {
          *v84 = v75;
          v17 = v84;
        }
        v84 = v17 + 1;
      }
      v19 = (_QWORD *)sub_317E470(v18);
      if ( v19 )
      {
        if ( v67 )
          break;
      }
      v20 = 0;
LABEL_49:
      v42 = sub_317E460(v75);
      v44 = v43;
      v73 = (int *)v42;
      v45 = (int *)sub_317E460(v69);
      sub_26C4ED0(v12, v45, v46, v73, v44, v20);
      v14 = sub_220EEE0(v14);
      if ( v68 == v14 )
        goto LABEL_50;
    }
    if ( unk_4F838D3 )
    {
      v20 = v19[8];
      if ( v20 )
      {
LABEL_39:
        v76 = sub_317E640(v75);
        v33 = *(_QWORD **)(v67 + 168);
        if ( v33 && (v34 = sub_C1BA30(v33, (__int64)&v76)) != 0 )
          v35 = (unsigned int *)(v34 + 16);
        else
          v35 = (unsigned int *)&v76;
        v36 = sub_26C2A80(v67 + 72, v35);
        if ( v36 != v67 + 80 )
        {
          v37 = v19[3];
          v38 = (_QWORD *)(v36 + 48);
          v77[0] = v19[2];
          v77[1] = v37;
          v72 = (int *)v77[0];
          if ( v77[0] )
          {
            v64 = v37;
            sub_C7D030(v88);
            sub_C7D280(v88, v72, v64);
            sub_C7D290(v88, &v78);
            v37 = v78;
            v38 = (_QWORD *)(v36 + 48);
          }
          v39 = sub_C1CC80(v38, v37 % *(_QWORD *)(v36 + 56), (__int64)v77, v37);
          if ( v39 )
          {
            v40 = *v39;
            if ( v40 )
            {
              v41 = *(_QWORD *)(v40 + 24);
              if ( v20 < v41 )
                v20 = v41;
            }
          }
        }
        goto LABEL_49;
      }
    }
    v21 = v19[20];
    if ( v19[14] )
    {
      v22 = v19[12];
      if ( !v21
        || (v23 = v19[18], v24 = *(_DWORD *)(v23 + 32), *(_DWORD *)(v22 + 32) < v24)
        || *(_DWORD *)(v22 + 32) == v24 && *(_DWORD *)(v22 + 36) < *(_DWORD *)(v23 + 36) )
      {
        v20 = *(_QWORD *)(v22 + 40);
LABEL_38:
        if ( v20 )
          goto LABEL_39;
        goto LABEL_58;
      }
    }
    else
    {
      if ( !v21 )
        goto LABEL_58;
      v23 = v19[18];
    }
    v71 = (_QWORD *)(v23 + 48);
    if ( *(_QWORD *)(v23 + 64) != v23 + 48 )
    {
      v63 = v19;
      v20 = 0;
      v62 = v14;
      v65 = unk_4F838D3;
      v61 = v12;
      v25 = *(_QWORD **)(v23 + 64);
      while ( 1 )
      {
        if ( v65 )
        {
          v26 = v25[14];
          if ( v26 )
            goto LABEL_36;
        }
        v27 = v25[26];
        if ( v25[20] )
        {
          v28 = v25[18];
          if ( !v27
            || (v29 = v25[24], v30 = *(_DWORD *)(v29 + 32), *(_DWORD *)(v28 + 32) < v30)
            || *(_DWORD *)(v28 + 32) == v30 && *(_DWORD *)(v28 + 36) < *(_DWORD *)(v29 + 36) )
          {
            v26 = *(_QWORD *)(v28 + 40);
            goto LABEL_35;
          }
        }
        else
        {
          if ( !v27 )
          {
LABEL_60:
            v26 = v25[13] != 0;
            goto LABEL_36;
          }
          v29 = v25[24];
        }
        v31 = *(_QWORD *)(v29 + 64);
        v32 = v29 + 48;
        if ( v31 == v32 )
          goto LABEL_60;
        v26 = 0;
        do
        {
          v26 += sub_EF9210((_QWORD *)(v31 + 48));
          v31 = sub_220EF30(v31);
        }
        while ( v32 != v31 );
LABEL_35:
        if ( !v26 )
          goto LABEL_60;
LABEL_36:
        v20 += v26;
        v25 = (_QWORD *)sub_220EF30((__int64)v25);
        if ( v71 == v25 )
        {
          v19 = v63;
          v14 = v62;
          v12 = v61;
          goto LABEL_38;
        }
      }
    }
LABEL_58:
    v20 = v19[7] != 0;
    goto LABEL_39;
  }
LABEL_52:
  if ( a3 )
  {
    if ( *(_DWORD *)(v3 + 104) )
    {
      v48 = *(_QWORD **)(v3 + 96);
      v49 = 2LL * *(unsigned int *)(v3 + 112);
      v50 = &v48[v49];
      v51 = &v48[v49];
      if ( v48 != &v48[v49] )
      {
        while ( 1 )
        {
          v52 = v48;
          if ( *v48 <= 0xFFFFFFFFFFFFFFFDLL )
            break;
          v48 += 2;
          if ( v50 == v48 )
            goto LABEL_53;
        }
        if ( v51 != v48 )
        {
          v53 = v51;
          do
          {
            v54 = v52[1];
            v55 = *(_QWORD *)(v54 + 40);
            v56 = (_QWORD *)(v54 + 24);
            v70 = v54;
            if ( v55 != v54 + 24 )
            {
              do
              {
                while ( 1 )
                {
                  v57 = *(_QWORD *)(v55 + 48);
                  v58 = (int *)v55;
                  v59 = sub_220EF30(v55);
                  v55 = v59;
                  if ( a3 >= v57 )
                    break;
                  if ( v56 == (_QWORD *)v59 )
                    goto LABEL_80;
                }
                v74 = (_QWORD *)v59;
                v60 = sub_220F330(v58, v56);
                j_j___libc_free_0((unsigned __int64)v60);
                v55 = (__int64)v74;
                --*(_QWORD *)(v70 + 56);
              }
              while ( v56 != v74 );
            }
LABEL_80:
            v52 += 2;
            if ( v52 == v53 )
              break;
            while ( *v52 > 0xFFFFFFFFFFFFFFFDLL )
            {
              v52 += 2;
              if ( v53 == v52 )
                goto LABEL_53;
            }
          }
          while ( v53 != v52 );
        }
      }
    }
  }
LABEL_53:
  sub_26C2C00((unsigned __int64 *)v79);
}
