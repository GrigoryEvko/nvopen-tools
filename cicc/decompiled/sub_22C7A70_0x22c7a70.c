// Function: sub_22C7A70
// Address: 0x22c7a70
//
__int64 __fastcall sub_22C7A70(__int64 a1, unsigned __int64 a2, unsigned __int8 *a3, __int64 a4)
{
  __int64 v6; // rax
  int v7; // edx
  __int64 v8; // r12
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r15
  int v13; // r15d
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 *v17; // r14
  __int64 v18; // r15
  __int64 v19; // rdi
  unsigned __int64 *v20; // rcx
  __int64 v21; // rsi
  int v22; // edx
  __int64 v23; // rsi
  unsigned int v24; // edx
  unsigned int v25; // edx
  __int64 v26; // rax
  char *v27; // r12
  __int64 v28; // rbx
  unsigned __int64 v29; // r12
  unsigned __int64 v30; // rdi
  unsigned __int64 *v31; // [rsp+18h] [rbp-158h]
  __int64 *v33; // [rsp+30h] [rbp-140h]
  __int64 v35[2]; // [rsp+40h] [rbp-130h] BYREF
  __int64 v36; // [rsp+50h] [rbp-120h] BYREF
  _WORD v37[24]; // [rsp+60h] [rbp-110h] BYREF
  unsigned __int8 v38[48]; // [rsp+90h] [rbp-E0h] BYREF
  unsigned __int64 v39; // [rsp+C0h] [rbp-B0h] BYREF
  unsigned int v40; // [rsp+C8h] [rbp-A8h]
  unsigned __int64 v41; // [rsp+D0h] [rbp-A0h]
  unsigned int v42; // [rsp+D8h] [rbp-98h]
  char v43; // [rsp+E0h] [rbp-90h]
  __int64 v44; // [rsp+F0h] [rbp-80h] BYREF
  __int64 v45; // [rsp+F8h] [rbp-78h]
  _BYTE v46[112]; // [rsp+100h] [rbp-70h] BYREF

  sub_22C07D0(v37, a3);
  v6 = *((_QWORD *)a3 - 4);
  if ( !v6 || *(_BYTE *)v6 || *(_QWORD *)(v6 + 24) != *((_QWORD *)a3 + 10) )
    BUG();
  if ( !sub_AB4EB0(*(_DWORD *)(v6 + 36)) )
  {
    sub_22C0650(a1, (unsigned __int8 *)v37);
    *(_BYTE *)(a1 + 40) = 1;
    goto LABEL_10;
  }
  v7 = *a3;
  v44 = (__int64)v46;
  v45 = 0x200000000LL;
  if ( v7 == 40 )
  {
    v8 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a3);
  }
  else
  {
    v8 = -32;
    if ( v7 != 85 )
    {
      v8 = -96;
      if ( v7 != 34 )
        BUG();
    }
  }
  if ( (a3[7] & 0x80u) != 0 )
  {
    v10 = sub_BD2BC0((__int64)a3);
    v12 = v10 + v11;
    if ( (a3[7] & 0x80u) == 0 )
    {
      if ( (unsigned int)(v12 >> 4) )
        goto LABEL_62;
    }
    else if ( (unsigned int)((v12 - sub_BD2BC0((__int64)a3)) >> 4) )
    {
      if ( (a3[7] & 0x80u) != 0 )
      {
        v13 = *(_DWORD *)(sub_BD2BC0((__int64)a3) + 8);
        if ( (a3[7] & 0x80u) == 0 )
          BUG();
        v14 = sub_BD2BC0((__int64)a3);
        v8 -= 32LL * (unsigned int)(*(_DWORD *)(v14 + v15 - 4) - v13);
        goto LABEL_18;
      }
LABEL_62:
      BUG();
    }
  }
LABEL_18:
  v33 = (__int64 *)&a3[v8];
  if ( &a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)] == &a3[v8] )
  {
LABEL_37:
    v26 = *((_QWORD *)a3 - 4);
    if ( !v26 || *(_BYTE *)v26 || *(_QWORD *)(v26 + 24) != *((_QWORD *)a3 + 10) )
      BUG();
    sub_ABD750((__int64)v35, *(_DWORD *)(v26 + 36), v44);
    sub_22C06B0((__int64)v38, (__int64)v35, 0);
    sub_22EACA0(&v39, v38, v37);
    sub_22C0650(a1, (unsigned __int8 *)&v39);
    *(_BYTE *)(a1 + 40) = 1;
    sub_22C0090((unsigned __int8 *)&v39);
    sub_22C0090(v38);
    sub_969240(&v36);
    sub_969240(v35);
  }
  else
  {
    v16 = a4;
    v17 = (__int64 *)&a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
    v18 = v16;
    sub_22C7770((__int64)&v39, a2, *v17, (__int64)a3, v16);
    while ( v43 )
    {
      v19 = (unsigned int)v45;
      v20 = &v39;
      v21 = v44;
      v22 = v45;
      if ( (unsigned __int64)(unsigned int)v45 + 1 > HIDWORD(v45) )
      {
        if ( v44 > (unsigned __int64)&v39 || (unsigned __int64)&v39 >= v44 + 32 * (unsigned __int64)(unsigned int)v45 )
        {
          sub_9D5330((__int64)&v44, (unsigned int)v45 + 1LL);
          v19 = (unsigned int)v45;
          v21 = v44;
          v20 = &v39;
          v22 = v45;
        }
        else
        {
          v27 = (char *)&v39 - v44;
          sub_9D5330((__int64)&v44, (unsigned int)v45 + 1LL);
          v21 = v44;
          v19 = (unsigned int)v45;
          v20 = (unsigned __int64 *)&v27[v44];
          v22 = v45;
        }
      }
      v23 = 32 * v19 + v21;
      if ( v23 )
      {
        v24 = *((_DWORD *)v20 + 2);
        *(_DWORD *)(v23 + 8) = v24;
        if ( v24 > 0x40 )
        {
          v31 = v20;
          sub_C43780(v23, (const void **)v20);
          v20 = v31;
        }
        else
        {
          *(_QWORD *)v23 = *v20;
        }
        v25 = *((_DWORD *)v20 + 6);
        *(_DWORD *)(v23 + 24) = v25;
        if ( v25 > 0x40 )
          sub_C43780(v23 + 16, (const void **)v20 + 2);
        else
          *(_QWORD *)(v23 + 16) = v20[2];
        v22 = v45;
      }
      LODWORD(v45) = v22 + 1;
      if ( !v43 )
        goto LABEL_20;
      v43 = 0;
      if ( v42 > 0x40 && v41 )
        j_j___libc_free_0_0(v41);
      if ( v40 > 0x40 && v39 )
      {
        j_j___libc_free_0_0(v39);
        v17 += 4;
        if ( v33 == v17 )
          goto LABEL_37;
      }
      else
      {
LABEL_20:
        v17 += 4;
        if ( v33 == v17 )
          goto LABEL_37;
      }
      sub_22C7770((__int64)&v39, a2, *v17, (__int64)a3, v18);
    }
    *(_BYTE *)(a1 + 40) = 0;
  }
  v28 = v44;
  v29 = v44 + 32LL * (unsigned int)v45;
  if ( v44 != v29 )
  {
    do
    {
      v29 -= 32LL;
      if ( *(_DWORD *)(v29 + 24) > 0x40u )
      {
        v30 = *(_QWORD *)(v29 + 16);
        if ( v30 )
          j_j___libc_free_0_0(v30);
      }
      if ( *(_DWORD *)(v29 + 8) > 0x40u && *(_QWORD *)v29 )
        j_j___libc_free_0_0(*(_QWORD *)v29);
    }
    while ( v28 != v29 );
    v29 = v44;
  }
  if ( (_BYTE *)v29 != v46 )
    _libc_free(v29);
LABEL_10:
  sub_22C0090((unsigned __int8 *)v37);
  return a1;
}
