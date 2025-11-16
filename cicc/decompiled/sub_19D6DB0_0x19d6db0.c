// Function: sub_19D6DB0
// Address: 0x19d6db0
//
__int64 __fastcall sub_19D6DB0(__int64 *a1)
{
  __int64 *v1; // r12
  __int64 v2; // rbx
  int v3; // eax
  int v4; // eax
  int v5; // eax
  __int64 v6; // r13
  __int64 v7; // rax
  bool v8; // cc
  __int64 v9; // rdi
  int v10; // eax
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdi
  const char *v14; // rax
  __int64 v15; // rdi
  const char *v16; // r15
  size_t v17; // rdx
  size_t v18; // r13
  size_t v19; // rdx
  const char *v20; // rsi
  size_t v21; // r14
  __int64 v22; // rdi
  __int64 v23; // rdi
  __int64 v25; // rcx
  unsigned __int64 v26; // rdx
  int v27; // eax
  __int64 v28; // [rsp+0h] [rbp-F0h]
  unsigned int v29; // [rsp+8h] [rbp-E8h]
  int v30; // [rsp+Ch] [rbp-E4h]
  __int64 v31; // [rsp+10h] [rbp-E0h]
  __int64 v32; // [rsp+18h] [rbp-D8h]
  __int64 v33; // [rsp+20h] [rbp-D0h]
  __int64 v34; // [rsp+28h] [rbp-C8h]
  int v35; // [rsp+30h] [rbp-C0h]
  char v36; // [rsp+37h] [rbp-B9h]
  __int64 v37; // [rsp+38h] [rbp-B8h]
  __int64 v38; // [rsp+40h] [rbp-B0h]
  __int64 v39; // [rsp+48h] [rbp-A8h]
  __int64 v40; // [rsp+80h] [rbp-70h] BYREF
  int v41; // [rsp+88h] [rbp-68h]
  __int64 v42; // [rsp+90h] [rbp-60h]
  __int64 v43; // [rsp+98h] [rbp-58h]
  __int64 v44; // [rsp+A0h] [rbp-50h]
  int v45; // [rsp+A8h] [rbp-48h]
  unsigned int v46; // [rsp+B0h] [rbp-40h]

  v1 = a1;
  v2 = a1[4];
  v39 = *a1;
  v38 = a1[1];
  v37 = a1[2];
  v36 = *((_BYTE *)a1 + 24);
  v34 = a1[5];
  v3 = *((_DWORD *)a1 + 14);
  *((_DWORD *)a1 + 14) = 0;
  v35 = v3;
  v41 = v3;
  v33 = a1[6];
  v40 = v33;
  v32 = a1[8];
  v42 = v32;
  v31 = a1[9];
  v43 = v31;
  v4 = *((_DWORD *)a1 + 22);
  *((_DWORD *)a1 + 22) = 0;
  v30 = v4;
  v45 = v4;
  v28 = a1[10];
  v44 = v28;
  v29 = *((_DWORD *)a1 + 24);
  v46 = v29;
  while ( 1 )
  {
    if ( v2 )
      v13 = *(_QWORD *)(v2 - 24LL * (*(_DWORD *)(v2 + 20) & 0xFFFFFFF));
    else
      v13 = 0;
    v14 = sub_1649960(v13);
    v15 = *(v1 - 9);
    v16 = v14;
    v18 = v17;
    if ( v15 )
      v15 = *(_QWORD *)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
    v20 = sub_1649960(v15);
    v21 = v19;
    if ( v18 > v19 )
    {
      if ( !v19 )
        goto LABEL_22;
      v5 = memcmp(v16, v20, v19);
      if ( v5 )
      {
LABEL_21:
        if ( v5 >= 0 )
          goto LABEL_22;
      }
      else
      {
LABEL_5:
        if ( v18 >= v21 )
          goto LABEL_22;
      }
      v6 = *(v1 - 9);
      goto LABEL_7;
    }
    if ( v18 )
    {
      v5 = memcmp(v16, v20, v18);
      if ( v5 )
        goto LABEL_21;
    }
    if ( v18 != v21 )
      goto LABEL_5;
    v6 = *(v1 - 9);
    if ( v2 )
    {
      v25 = *(_QWORD *)(v2 - 24LL * (*(_DWORD *)(v2 + 20) & 0xFFFFFFF));
      if ( v6 )
      {
        if ( *(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF)) == v25 )
          break;
      }
      else if ( !v25 )
      {
        break;
      }
      v26 = *(_QWORD *)(v2 - 24LL * (*(_DWORD *)(v2 + 20) & 0xFFFFFFF));
      if ( !v6 )
        goto LABEL_22;
      v27 = *(_DWORD *)(v6 + 20);
      goto LABEL_34;
    }
    if ( !v6 )
      break;
    v27 = *(_DWORD *)(v6 + 20);
    if ( !*(_QWORD *)(v6 - 24LL * (v27 & 0xFFFFFFF)) )
      break;
    v26 = 0;
LABEL_34:
    if ( *(_QWORD *)(v6 - 24LL * (v27 & 0xFFFFFFF)) <= v26 )
      goto LABEL_22;
LABEL_7:
    v7 = *(v1 - 13);
    v8 = *((_DWORD *)v1 + 14) <= 0x40u;
    v1[4] = v6;
    *v1 = v7;
    v1[1] = *(v1 - 12);
    v1[2] = *(v1 - 11);
    *((_BYTE *)v1 + 24) = *((_BYTE *)v1 - 80);
    v1[5] = *(v1 - 8);
    if ( !v8 )
    {
      v9 = v1[6];
      if ( v9 )
        j_j___libc_free_0_0(v9);
    }
    v8 = *((_DWORD *)v1 + 22) <= 0x40u;
    v1[6] = *(v1 - 7);
    v10 = *((_DWORD *)v1 - 12);
    *((_DWORD *)v1 - 12) = 0;
    *((_DWORD *)v1 + 14) = v10;
    v1[8] = *(v1 - 5);
    v1[9] = *(v1 - 4);
    if ( !v8 )
    {
      v11 = v1[10];
      if ( v11 )
        j_j___libc_free_0_0(v11);
    }
    v12 = *(v1 - 3);
    v1 -= 13;
    v1[23] = v12;
    LODWORD(v12) = *((_DWORD *)v1 + 22);
    *((_DWORD *)v1 + 22) = 0;
    *((_DWORD *)v1 + 48) = v12;
    *((_DWORD *)v1 + 50) = *((_DWORD *)v1 + 24);
  }
  if ( (int)sub_16AEA10((__int64)&v40, (__int64)(v1 - 7)) < 0 )
    goto LABEL_7;
LABEL_22:
  v8 = *((_DWORD *)v1 + 14) <= 0x40u;
  v1[4] = v2;
  *v1 = v39;
  v1[1] = v38;
  v1[2] = v37;
  *((_BYTE *)v1 + 24) = v36;
  v1[5] = v34;
  if ( !v8 )
  {
    v22 = v1[6];
    if ( v22 )
      j_j___libc_free_0_0(v22);
  }
  v8 = *((_DWORD *)v1 + 22) <= 0x40u;
  v1[6] = v33;
  *((_DWORD *)v1 + 14) = v35;
  v1[8] = v32;
  v1[9] = v31;
  if ( !v8 )
  {
    v23 = v1[10];
    if ( v23 )
      j_j___libc_free_0_0(v23);
  }
  v1[10] = v28;
  *((_DWORD *)v1 + 22) = v30;
  *((_DWORD *)v1 + 24) = v29;
  return v29;
}
