// Function: sub_ED2230
// Address: 0xed2230
//
void __fastcall sub_ED2230(__int64 **a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, unsigned int a6, int a7)
{
  __int64 *v7; // r12
  __int64 v11; // rax
  __int64 v12; // r9
  __int64 v13; // rdx
  unsigned __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // r9
  __int64 v21; // rdx
  unsigned __int64 v22; // r8
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rax
  __int64 v28; // r9
  __int64 v29; // rdx
  unsigned __int64 v30; // r8
  __int64 v31; // r14
  __int64 *v32; // rdx
  __int64 *v33; // rbx
  __int64 v34; // r14
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 v40; // r8
  __int64 v41; // rdx
  unsigned __int64 v42; // r9
  __int64 v43; // r14
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // rax
  __int64 v49; // r8
  __int64 v50; // rdx
  unsigned __int64 v51; // r9
  __int64 v52; // rax
  __int64 v53; // [rsp+8h] [rbp-98h]
  __int64 v54; // [rsp+8h] [rbp-98h]
  __int64 v55; // [rsp+8h] [rbp-98h]
  __int64 *v57; // [rsp+28h] [rbp-78h]
  __int64 v58; // [rsp+28h] [rbp-78h]
  __int64 v59; // [rsp+28h] [rbp-78h]
  __int64 *v60; // [rsp+38h] [rbp-68h] BYREF
  __int64 *v61; // [rsp+40h] [rbp-60h] BYREF
  __int64 v62; // [rsp+48h] [rbp-58h]
  _BYTE v63[80]; // [rsp+50h] [rbp-50h] BYREF

  if ( a4 )
  {
    v7 = *a1;
    v60 = *a1;
    v61 = (__int64 *)v63;
    v62 = 0x300000000LL;
    v11 = sub_B8C130(&v60, (__int64)"VP", 2);
    v13 = (unsigned int)v62;
    v14 = (unsigned int)v62 + 1LL;
    if ( v14 > HIDWORD(v62) )
    {
      v55 = v11;
      sub_C8D5F0((__int64)&v61, v63, (unsigned int)v62 + 1LL, 8u, v14, v12);
      v13 = (unsigned int)v62;
      v11 = v55;
    }
    v61[v13] = v11;
    LODWORD(v62) = v62 + 1;
    v15 = sub_BCB2D0(v7);
    v16 = sub_ACD640(v15, a6, 0);
    v19 = sub_B8C140((__int64)&v60, v16, v17, v18);
    v21 = (unsigned int)v62;
    v22 = (unsigned int)v62 + 1LL;
    if ( v22 > HIDWORD(v62) )
    {
      v58 = v19;
      sub_C8D5F0((__int64)&v61, v63, (unsigned int)v62 + 1LL, 8u, v22, v20);
      v21 = (unsigned int)v62;
      v19 = v58;
    }
    v61[v21] = v19;
    LODWORD(v62) = v62 + 1;
    v23 = sub_BCB2E0(v7);
    v24 = sub_ACD640(v23, a5, 0);
    v27 = sub_B8C140((__int64)&v60, v24, v25, v26);
    v29 = (unsigned int)v62;
    v30 = (unsigned int)v62 + 1LL;
    if ( v30 > HIDWORD(v62) )
    {
      v59 = v27;
      sub_C8D5F0((__int64)&v61, v63, (unsigned int)v62 + 1LL, 8u, v30, v28);
      v29 = (unsigned int)v62;
      v27 = v59;
    }
    v31 = 2 * a4;
    v61[v29] = v27;
    v32 = (__int64 *)(unsigned int)(v62 + 1);
    LODWORD(v62) = v62 + 1;
    v57 = &a3[v31];
    if ( a3 != &a3[v31] )
    {
      v33 = &a3[2 * (unsigned int)(a7 - 1)];
      do
      {
        v43 = *a3;
        v44 = sub_BCB2E0(v7);
        v45 = sub_ACD640(v44, v43, 0);
        v48 = sub_B8C140((__int64)&v60, v45, v46, v47);
        v50 = (unsigned int)v62;
        v51 = (unsigned int)v62 + 1LL;
        if ( v51 > HIDWORD(v62) )
        {
          v53 = v48;
          sub_C8D5F0((__int64)&v61, v63, (unsigned int)v62 + 1LL, 8u, v49, v51);
          v50 = (unsigned int)v62;
          v48 = v53;
        }
        v61[v50] = v48;
        v34 = a3[1];
        LODWORD(v62) = v62 + 1;
        v35 = sub_BCB2E0(v7);
        v36 = sub_ACD640(v35, v34, 0);
        v39 = sub_B8C140((__int64)&v60, v36, v37, v38);
        v41 = (unsigned int)v62;
        v42 = (unsigned int)v62 + 1LL;
        if ( v42 > HIDWORD(v62) )
        {
          v54 = v39;
          sub_C8D5F0((__int64)&v61, v63, (unsigned int)v62 + 1LL, 8u, v40, v42);
          v41 = (unsigned int)v62;
          v39 = v54;
        }
        v61[v41] = v39;
        v32 = (__int64 *)(unsigned int)(v62 + 1);
        LODWORD(v62) = v62 + 1;
        if ( v33 == a3 )
          break;
        a3 += 2;
      }
      while ( v57 != a3 );
    }
    v52 = sub_B9C770(v7, v61, v32, 0, 1);
    sub_B99FD0(a2, 2u, v52);
    if ( v61 != (__int64 *)v63 )
      _libc_free(v61, 2);
  }
}
