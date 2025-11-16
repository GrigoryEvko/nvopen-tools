// Function: sub_34A86E0
// Address: 0x34a86e0
//
unsigned __int64 __fastcall sub_34A86E0(unsigned int *a1, unsigned __int64 a2, unsigned __int64 a3, int a4)
{
  unsigned int *v5; // rbx
  unsigned int v6; // eax
  __int64 v7; // rsi
  __int64 v8; // rsi
  int v9; // r14d
  __int64 v10; // r13
  unsigned int v11; // r15d
  unsigned __int64 result; // rax
  bool v13; // r14
  unsigned int v14; // esi
  __int64 v15; // rdx
  unsigned __int64 *v16; // rcx
  int v17; // esi
  unsigned __int64 *v18; // r14
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // r14
  __int64 v22; // rax
  __int64 v23; // r14
  __int64 v24; // rax
  unsigned __int64 v25; // r14
  __int64 v26; // rdi
  __int64 v27; // r13
  __int64 v28; // rax
  unsigned int v29; // r9d
  unsigned int v30; // eax
  unsigned int v31; // edx
  __int64 v32; // r15
  unsigned int v33; // r14d
  int v34; // r14d
  __int64 v35; // rax
  unsigned int v36; // edx
  unsigned int v37; // r9d
  char v38; // si
  unsigned __int64 v39; // rax
  __int64 v40; // r15
  int v41; // esi
  bool v42; // zf
  unsigned int v43; // edx
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rax
  _QWORD *v47; // rdx
  unsigned int v48; // r14d
  __int64 *v49; // r13
  __int64 v50; // rbx
  int v51; // r9d
  __int64 v52; // rdi
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // rax
  unsigned __int64 *v56; // rax
  unsigned int i; // r15d
  __int64 v58; // rax
  unsigned int v59; // edx
  unsigned int v60; // eax
  __int64 v61; // rsi
  unsigned __int64 v62; // rcx
  __int64 v63; // rax
  __int64 v64; // [rsp+18h] [rbp-A8h]
  unsigned int v65; // [rsp+20h] [rbp-A0h]
  char v66; // [rsp+20h] [rbp-A0h]
  char v67; // [rsp+24h] [rbp-9Ch]
  __int64 v68; // [rsp+28h] [rbp-98h]
  __int64 v69; // [rsp+30h] [rbp-90h]
  unsigned int v70; // [rsp+38h] [rbp-88h]
  unsigned int v71; // [rsp+38h] [rbp-88h]
  int v72; // [rsp+3Ch] [rbp-84h]
  unsigned int v73; // [rsp+3Ch] [rbp-84h]
  unsigned int v74; // [rsp+3Ch] [rbp-84h]
  unsigned int v75; // [rsp+3Ch] [rbp-84h]
  unsigned int v78[4]; // [rsp+50h] [rbp-70h] BYREF
  _DWORD v79[4]; // [rsp+60h] [rbp-60h] BYREF
  _QWORD v80[10]; // [rsp+70h] [rbp-50h] BYREF

  v5 = a1 + 2;
  v6 = a1[4];
  if ( !v6 || (v7 = *((_QWORD *)a1 + 1), *(_DWORD *)(v7 + 12) >= *(_DWORD *)(v7 + 8)) )
  {
    v72 = a4;
    v21 = 16LL * *(unsigned int *)(*(_QWORD *)a1 + 192LL);
    sub_F03AD0(v5, *(_DWORD *)(*(_QWORD *)a1 + 192LL));
    a4 = v72;
    ++*(_DWORD *)(*((_QWORD *)a1 + 1) + v21 + 12);
    v6 = a1[4];
    v7 = *((_QWORD *)a1 + 1);
  }
  v8 = v7 + 16LL * v6 - 16;
  v9 = *(_DWORD *)(v8 + 12);
  v10 = *(_QWORD *)v8;
  if ( !v9 && *(_QWORD *)v10 > a2 )
  {
    v73 = a4;
    v22 = sub_F03A30((__int64 *)v5, v6 - 1);
    LOBYTE(a4) = v73;
    if ( v22 )
    {
      v23 = v22;
      v24 = v22 & 0x3F;
      v25 = v23 & 0xFFFFFFFFFFFFFFC0LL;
      v26 = a1[4];
      v8 = *((_QWORD *)a1 + 1) + 16 * v26 - 16;
      v10 = *(_QWORD *)v8;
      if ( *(_BYTE *)(v25 + v24 + 176) == (_BYTE)v73 && (v18 = (unsigned __int64 *)(16 * v24 + v25), v18[1] + 1 == a2) )
      {
        sub_F03AD0(v5, v26 - 1);
        if ( *(_QWORD *)v10 > a3 )
        {
          if ( *(_BYTE *)(v10 + 176) != (_BYTE)v73 || (v19 = a3 + 1, *(_QWORD *)v10 != a3 + 1) )
          {
            result = a3;
            v18[1] = a3;
            v17 = a1[4] - 1;
            if ( a1[4] != 1 )
              return sub_349D820((__int64)a1, v17, a3);
            return result;
          }
        }
        a2 = *v18;
        sub_34A2FF0((__int64)a1, 0, v19, v73, v20);
        LOBYTE(a4) = v73;
        v8 = *((_QWORD *)a1 + 1) + 16LL * a1[4] - 16;
        v9 = *(_DWORD *)(v8 + 12);
        v10 = *(_QWORD *)v8;
      }
      else
      {
        v9 = *(_DWORD *)(v8 + 12);
      }
    }
    else
    {
      **(_QWORD **)a1 = a2;
      v8 = *((_QWORD *)a1 + 1) + 16LL * a1[4] - 16;
      v9 = *(_DWORD *)(v8 + 12);
      v10 = *(_QWORD *)v8;
    }
  }
  v11 = *(_DWORD *)(v8 + 8);
  v67 = a4;
  result = sub_34A32D0(v10, (unsigned int *)(v8 + 12), v11, a2, a3, a4);
  v13 = v11 == v9;
  if ( (unsigned int)result <= 0xB )
    goto LABEL_6;
  v27 = a1[4] - 1;
  v74 = *(_DWORD *)(*((_QWORD *)a1 + 1) + 16 * v27 + 12);
  v28 = sub_F03A30((__int64 *)v5, a1[4] - 1);
  v29 = v74;
  v69 = v28;
  if ( v28 )
  {
    v31 = 1;
    v75 = 2;
    v80[0] = v28 & 0xFFFFFFFFFFFFFFC0LL;
    v30 = (v28 & 0x3F) + 1;
    v78[0] = v30;
    v29 += v30;
  }
  else
  {
    v75 = 1;
    v30 = 0;
    v31 = 0;
  }
  v32 = *((_QWORD *)a1 + 1) + 16 * v27;
  v33 = *(_DWORD *)(v32 + 8);
  v65 = v29;
  v70 = v31;
  v78[v31] = v33;
  v34 = v30 + v33;
  v68 = v31;
  v80[v31] = *(_QWORD *)v32;
  v35 = sub_F03C90((__int64 *)v5, v27);
  v36 = v70;
  v37 = v65;
  if ( v35 )
  {
    v38 = v35;
    v39 = v35 & 0xFFFFFFFFFFFFFFC0LL;
    v40 = v75;
    v71 = v70 + 2;
    v41 = (v38 & 0x3F) + 1;
    v34 += v41;
    v42 = v36 == 0;
    v43 = 22;
    v78[v75] = v41;
    if ( !v42 )
      v43 = 33;
    v80[v75] = v39;
    if ( v34 + 1 > v43 )
      goto LABEL_27;
LABEL_48:
    v66 = 0;
    v75 = 0;
    goto LABEL_32;
  }
  v60 = 11;
  if ( v75 != 1 )
    v60 = 22;
  if ( v34 + 1 <= v60 )
  {
    v71 = v75;
    goto LABEL_48;
  }
  if ( v75 != 1 )
  {
    v75 = v70;
    v41 = v78[v68];
    v40 = v68;
    v71 = 2;
    v39 = v80[v68];
LABEL_27:
    v44 = v71++;
    goto LABEL_28;
  }
  v41 = v78[1];
  v39 = v80[1];
  v44 = 1;
  v71 = 2;
  v40 = 1;
LABEL_28:
  v80[v44] = v39;
  v45 = *(_QWORD *)a1;
  v78[v44] = v41;
  v78[v40] = 0;
  v46 = *(_QWORD *)(v45 + 200);
  v47 = *(_QWORD **)v46;
  if ( *(_QWORD *)v46 )
  {
    *(_QWORD *)v46 = *v47;
  }
  else
  {
    v61 = *(_QWORD *)(v46 + 8);
    *(_QWORD *)(v46 + 88) += 192LL;
    v62 = (v61 + 63) & 0xFFFFFFFFFFFFFFC0LL;
    if ( *(_QWORD *)(v46 + 16) >= v62 + 192 && v61 )
    {
      *(_QWORD *)(v46 + 8) = v62 + 192;
      if ( !v62 )
        goto LABEL_31;
      v47 = (_QWORD *)((v61 + 63) & 0xFFFFFFFFFFFFFFC0LL);
    }
    else
    {
      v63 = sub_9D1E70(v46 + 8, 192, 192, 6);
      v37 = v65;
      v47 = (_QWORD *)v63;
    }
  }
  memset(v47, 0, 0xC0u);
LABEL_31:
  v80[v40] = v47;
  v66 = 1;
LABEL_32:
  v64 = sub_F03E60(v71, v34, 11, (__int64)v78, (__int64)v79, v37, 1u);
  sub_34A76E0((__int64)v80, v71, v78, (__int64)v79);
  if ( v69 )
    sub_F03AD0(v5, v27);
  v48 = v27;
  v49 = (__int64 *)v5;
  v50 = 0;
  while ( 1 )
  {
    v51 = v79[v50];
    v52 = (unsigned int)(v51 - 1);
    v53 = v80[v50];
    v54 = *(_QWORD *)(v53 + 16 * v52 + 8);
    if ( v75 == (_DWORD)v50 && v66 )
    {
      v48 += (unsigned __int8)sub_34A8450(a1, v48, v52 | v53 & 0xFFFFFFFFFFFFFFC0LL, v54);
      goto LABEL_38;
    }
    *(_DWORD *)(*((_QWORD *)a1 + 1) + 16LL * v48 + 8) = v51;
    if ( v48 )
      break;
LABEL_38:
    if ( v71 == ++v50 )
      goto LABEL_44;
LABEL_39:
    sub_F03D40(v49, v48);
  }
  ++v50;
  v55 = *((_QWORD *)a1 + 1) + 16LL * (v48 - 1);
  v56 = (unsigned __int64 *)(*(_QWORD *)v55 + 8LL * *(unsigned int *)(v55 + 12));
  *v56 = v52 | *v56 & 0xFFFFFFFFFFFFFFC0LL;
  sub_349D820((__int64)a1, v48, v54);
  if ( v71 != v50 )
    goto LABEL_39;
LABEL_44:
  for ( i = v71 - 1; i != (_DWORD)v64; --i )
    sub_F03AD0((unsigned int *)v49, v48);
  *(_DWORD *)(*((_QWORD *)a1 + 1) + 16LL * v48 + 12) = HIDWORD(v64);
  v58 = *((_QWORD *)a1 + 1) + 16LL * a1[4] - 16;
  v59 = *(_DWORD *)(v58 + 8);
  v13 = *(_DWORD *)(v58 + 12) == v59;
  result = sub_34A32D0(*(_QWORD *)v58, (unsigned int *)(v58 + 12), v59, a2, a3, v67);
LABEL_6:
  v14 = a1[4];
  *(_DWORD *)(*((_QWORD *)a1 + 1) + 16LL * (v14 - 1) + 8) = result;
  if ( v14 != 1 )
  {
    v15 = *((_QWORD *)a1 + 1) + 16LL * (v14 - 2);
    v16 = (unsigned __int64 *)(*(_QWORD *)v15 + 8LL * *(unsigned int *)(v15 + 12));
    result = *v16 & 0xFFFFFFFFFFFFFFC0LL | (unsigned int)(result - 1);
    *v16 = result;
  }
  if ( v13 )
  {
    v17 = a1[4] - 1;
    if ( a1[4] != 1 )
      return sub_349D820((__int64)a1, v17, a3);
  }
  return result;
}
