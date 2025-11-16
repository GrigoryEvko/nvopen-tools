// Function: sub_2CBF770
// Address: 0x2cbf770
//
__int64 __fastcall sub_2CBF770(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 **v10; // rdx
  __int64 v11; // rax
  __int64 *v12; // r13
  __int64 v13; // r8
  __int64 v14; // rbx
  int v15; // r13d
  __int64 v16; // r12
  __int64 v17; // r14
  __int64 *v18; // r15
  __int64 v19; // rsi
  _QWORD *v20; // rax
  _QWORD *v21; // r10
  int v22; // r9d
  __int64 v23; // r8
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // r8
  __int64 v31; // rax
  __int64 v32; // r8
  __int64 v33; // r10
  int v34; // edx
  int v35; // edx
  unsigned int v36; // ecx
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // rcx
  __int64 *v40; // rcx
  __int64 v41; // rdx
  __int64 v42; // r8
  __int64 *v43; // rax
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 *v47; // rdx
  __int64 v48; // rax
  __int64 v49; // [rsp+8h] [rbp-B8h]
  __int64 v50; // [rsp+10h] [rbp-B0h]
  __int64 *v51; // [rsp+18h] [rbp-A8h]
  __int64 *v52; // [rsp+20h] [rbp-A0h]
  __int64 v53; // [rsp+28h] [rbp-98h]
  __int64 v54; // [rsp+28h] [rbp-98h]
  __int64 v55; // [rsp+28h] [rbp-98h]
  __int64 v56; // [rsp+28h] [rbp-98h]
  __int64 v58; // [rsp+38h] [rbp-88h]
  __int64 *v59; // [rsp+40h] [rbp-80h]
  __int64 v60; // [rsp+48h] [rbp-78h] BYREF
  __int64 *v61; // [rsp+58h] [rbp-68h] BYREF
  const char *v62[4]; // [rsp+60h] [rbp-60h] BYREF
  char v63; // [rsp+80h] [rbp-40h]
  char v64; // [rsp+81h] [rbp-3Fh]

  v6 = *(_QWORD *)(a3 + 56);
  v60 = a1;
  v58 = a3 + 48;
  if ( v6 == a3 + 48 )
    return 1;
  while ( 2 )
  {
    if ( !v6 )
      BUG();
    if ( *(_BYTE *)(v6 - 24) != 84 )
      return 1;
    v10 = *(__int64 ***)(v6 - 32);
    v11 = 4LL * *(unsigned int *)(v6 + 48);
    if ( a4 == v10[v11] )
    {
      v12 = *v10;
      if ( *(_BYTE *)*v10 <= 0x1Cu )
        return 0;
    }
    else
    {
      if ( a4 != v10[v11 + 1] )
        BUG();
      v12 = v10[4];
      if ( *(_BYTE *)v12 <= 0x1Cu )
        return 0;
    }
    v13 = v12[2];
    if ( !v13 )
      goto LABEL_27;
    v49 = a6;
    v53 = v6;
    v14 = v60;
    v52 = v12;
    v15 = 0;
    v51 = a4;
    v16 = v13;
    v50 = a5;
    v17 = v60 + 56;
    do
    {
      while ( 1 )
      {
        v18 = *(__int64 **)(v16 + 24);
        if ( *(_BYTE *)v18 <= 0x1Cu )
          goto LABEL_15;
        v19 = v18[5];
        if ( *(_BYTE *)(v14 + 84) )
          break;
        if ( !sub_C8CA60(v17, v19) )
          goto LABEL_50;
LABEL_15:
        v16 = *(_QWORD *)(v16 + 8);
        if ( !v16 )
          goto LABEL_16;
      }
      v20 = *(_QWORD **)(v14 + 64);
      v21 = &v20[*(unsigned int *)(v14 + 76)];
      if ( v20 != v21 )
      {
        while ( v19 != *v20 )
        {
          if ( v21 == ++v20 )
            goto LABEL_50;
        }
        goto LABEL_15;
      }
LABEL_50:
      v16 = *(_QWORD *)(v16 + 8);
      v59 = v18;
      ++v15;
    }
    while ( v16 );
LABEL_16:
    v22 = v15;
    v23 = v16;
    v6 = v53;
    v12 = v52;
    a4 = v51;
    a5 = v50;
    a6 = v49;
    if ( v22 != 1 )
      goto LABEL_19;
    v24 = v59[5];
    if ( v24 == v49 )
      goto LABEL_40;
    if ( v24 != v50 )
    {
LABEL_19:
      v25 = v52[2];
      if ( v25 )
      {
        v26 = v52[2];
        while ( 1 )
        {
          v27 = *(_QWORD *)(v26 + 24);
          if ( *(_BYTE *)v27 > 0x1Cu && v50 == *(_QWORD *)(v27 + 40) )
            return 0;
          v26 = *(_QWORD *)(v26 + 8);
          if ( !v26 )
          {
            while ( 1 )
            {
              v28 = *(_QWORD *)(v25 + 24);
              if ( *(_BYTE *)v28 > 0x1Cu && v49 == *(_QWORD *)(v28 + 40) )
                return 0;
              v25 = *(_QWORD *)(v25 + 8);
              if ( !v25 )
                goto LABEL_27;
            }
          }
        }
      }
LABEL_27:
      v29 = v12[1];
      v30 = *(_QWORD *)(a5 + 56);
      v64 = 1;
      v54 = v30;
      v62[0] = "splitPhi";
      v63 = 3;
      v31 = sub_BD2DA0(80);
      v32 = v54;
      v33 = v31;
      if ( v31 )
      {
        v55 = v31;
        sub_B44260(v31, v29, 55, 0x8000000u, v32, 1u);
        *(_DWORD *)(v55 + 72) = 1;
        sub_BD6B50((unsigned __int8 *)v55, v62);
        sub_BD2A10(v55, *(_DWORD *)(v55 + 72), 1);
        v33 = v55;
      }
      v34 = *(_DWORD *)(v33 + 4) & 0x7FFFFFF;
      if ( v34 == *(_DWORD *)(v33 + 72) )
      {
        v56 = v33;
        sub_B48D90(v33);
        v33 = v56;
        v34 = *(_DWORD *)(v56 + 4) & 0x7FFFFFF;
      }
      v35 = (v34 + 1) & 0x7FFFFFF;
      v36 = v35 | *(_DWORD *)(v33 + 4) & 0xF8000000;
      v37 = *(_QWORD *)(v33 - 8) + 32LL * (unsigned int)(v35 - 1);
      *(_DWORD *)(v33 + 4) = v36;
      if ( *(_QWORD *)v37 )
      {
        v38 = *(_QWORD *)(v37 + 8);
        **(_QWORD **)(v37 + 16) = v38;
        if ( v38 )
          *(_QWORD *)(v38 + 16) = *(_QWORD *)(v37 + 16);
      }
      *(_QWORD *)v37 = v12;
      v39 = v12[2];
      *(_QWORD *)(v37 + 8) = v39;
      if ( v39 )
        *(_QWORD *)(v39 + 16) = v37 + 8;
      *(_QWORD *)(v37 + 16) = v12 + 2;
      v12[2] = v37;
      *(_QWORD *)(*(_QWORD *)(v33 - 8)
                + 32LL * *(unsigned int *)(v33 + 72)
                + 8LL * ((*(_DWORD *)(v33 + 4) & 0x7FFFFFFu) - 1)) = a4;
      v61 = (__int64 *)v33;
      v40 = *(__int64 **)(v6 - 32);
      v41 = 4LL * *(unsigned int *)(v6 + 48);
      if ( a4 == (__int64 *)v40[v41] )
      {
        v42 = v40[4];
      }
      else
      {
        v42 = 0;
        if ( a4 == (__int64 *)v40[v41 + 1] )
          v42 = *v40;
      }
      v43 = (__int64 *)sub_2CBF540(a6, *(_QWORD *)(v33 + 8), v33, a5, v42, a2);
      v62[0] = (const char *)&v61;
      v62[1] = (const char *)&v60;
      sub_BD79D0(v12, v43, (unsigned __int8 (__fastcall *)(__int64, __int64 *))sub_2CBF220, (__int64)v62);
LABEL_40:
      v6 = *(_QWORD *)(v6 + 8);
      if ( v58 == v6 )
        return 1;
      continue;
    }
    break;
  }
  v46 = v59[2];
  if ( !v46 )
  {
LABEL_56:
    v47 = *(__int64 **)(v53 - 32);
    v48 = 4LL * *(unsigned int *)(v53 + 48);
    if ( v51 == (__int64 *)v47[v48] )
    {
      v23 = v47[4];
    }
    else if ( v51 == (__int64 *)v47[v48 + 1] )
    {
      v23 = *v47;
    }
    v61 = (__int64 *)sub_2CBF540(v49, v59[1], (__int64)v59, v50, v23, a2);
    v62[0] = (const char *)&v61;
    sub_BD79D0(v59, v61, (unsigned __int8 (__fastcall *)(__int64, __int64 *))sub_2CBEC50, (__int64)v62);
    goto LABEL_40;
  }
  v45 = *(_QWORD *)(v46 + 24);
  if ( !*(_QWORD *)(v46 + 8) && v45 && v49 == *(_QWORD *)(v45 + 40) )
    goto LABEL_40;
  while ( *(_BYTE *)v45 <= 0x1Cu || v49 != *(_QWORD *)(v45 + 40) )
  {
    v46 = *(_QWORD *)(v46 + 8);
    if ( !v46 )
      goto LABEL_56;
    v45 = *(_QWORD *)(v46 + 24);
  }
  return 0;
}
