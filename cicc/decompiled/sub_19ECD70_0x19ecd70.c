// Function: sub_19ECD70
// Address: 0x19ecd70
//
__int64 __fastcall sub_19ECD70(__int64 a1, __int64 a2)
{
  int v4; // eax
  int v5; // ecx
  __int64 v6; // rsi
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // rdi
  int v10; // eax
  int v12; // eax
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 *v17; // rcx
  __int64 *v18; // rcx
  __int64 *v19; // r8
  __int64 *v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 *v26; // rax
  __int64 v27; // rax
  char v28; // al
  __int64 v29; // rdx
  unsigned int v30; // esi
  int v31; // eax
  int v32; // eax
  char v33; // al
  __int64 v34; // rdx
  int v35; // r8d
  unsigned int v36; // esi
  int v37; // eax
  int v38; // eax
  int v39; // [rsp+Ch] [rbp-D4h]
  __int64 v40; // [rsp+18h] [rbp-C8h] BYREF
  __int64 v41; // [rsp+20h] [rbp-C0h] BYREF
  int v42; // [rsp+28h] [rbp-B8h]
  _QWORD v43[4]; // [rsp+30h] [rbp-B0h] BYREF
  _QWORD v44[4]; // [rsp+50h] [rbp-90h] BYREF
  _QWORD v45[4]; // [rsp+70h] [rbp-70h] BYREF
  __int64 *v46; // [rsp+90h] [rbp-50h] BYREF
  __int64 *v47; // [rsp+98h] [rbp-48h]
  __int64 v48; // [rsp+A0h] [rbp-40h]
  __int64 v49; // [rsp+A8h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 2048);
  if ( v4 )
  {
    v5 = v4 - 1;
    v6 = *(_QWORD *)(a1 + 2032);
    v7 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
    {
LABEL_3:
      v10 = *((_DWORD *)v8 + 2);
      if ( v10 )
      {
        LOBYTE(a2) = v10 != 2;
        return (unsigned int)a2;
      }
    }
    else
    {
      v12 = 1;
      while ( v9 != -8 )
      {
        v35 = v12 + 1;
        v7 = v5 & (v12 + v7);
        v8 = (__int64 *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( a2 == *v8 )
          goto LABEL_3;
        v12 = v35;
      }
    }
  }
  if ( !(unsigned int)sub_19E5210(a1 + 360, a2) )
    sub_19EC800(a1 + 248, a2);
  v13 = *(_QWORD *)(a1 + 472) + 104LL * (unsigned int)sub_19E5210(a1 + 1320, a2);
  v14 = *(unsigned int *)(v13 + 28);
  if ( *(_DWORD *)(v13 + 28) - *(_DWORD *)(v13 + 32) == 1 )
  {
    v46 = (__int64 *)a2;
    LODWORD(v47) = 1;
    v33 = sub_19E1070(a1 + 2024, (__int64 *)&v46, v45);
    v34 = v45[0];
    if ( v33 )
    {
LABEL_40:
      LODWORD(a2) = 1;
      return (unsigned int)a2;
    }
    v36 = *(_DWORD *)(a1 + 2048);
    v37 = *(_DWORD *)(a1 + 2040);
    ++*(_QWORD *)(a1 + 2024);
    v38 = v37 + 1;
    if ( 4 * v38 >= 3 * v36 )
    {
      v36 *= 2;
    }
    else if ( v36 - *(_DWORD *)(a1 + 2044) - v38 > v36 >> 3 )
    {
LABEL_47:
      *(_DWORD *)(a1 + 2040) = v38;
      if ( *(_QWORD *)v34 != -8 )
        --*(_DWORD *)(a1 + 2044);
      *(_QWORD *)v34 = v46;
      *(_DWORD *)(v34 + 8) = (_DWORD)v47;
      goto LABEL_40;
    }
    sub_19E3FB0(a1 + 2024, v36);
    sub_19E1070(a1 + 2024, (__int64 *)&v46, v45);
    v34 = v45[0];
    v38 = *(_DWORD *)(a1 + 2040) + 1;
    goto LABEL_47;
  }
  v15 = *(_QWORD *)(v13 + 16);
  if ( v15 != *(_QWORD *)(v13 + 8) )
    v14 = *(unsigned int *)(v13 + 24);
  v45[0] = v15 + 8 * v14;
  v45[1] = v45[0];
  sub_19E4730((__int64)v45);
  v45[2] = v13;
  v45[3] = *(_QWORD *)v13;
  v16 = *(_QWORD *)(v13 + 16);
  if ( v16 == *(_QWORD *)(v13 + 8) )
    v17 = (__int64 *)(v16 + 8LL * *(unsigned int *)(v13 + 28));
  else
    v17 = (__int64 *)(v16 + 8LL * *(unsigned int *)(v13 + 24));
  v46 = *(__int64 **)(v13 + 16);
  v47 = v17;
  sub_19E4730((__int64)&v46);
  v48 = v13;
  v18 = v46;
  v19 = (__int64 *)v45[0];
  v20 = v47;
  v49 = *(_QWORD *)v13;
  if ( (__int64 *)v45[0] == v46 )
  {
LABEL_37:
    v39 = 1;
    LODWORD(a2) = 1;
  }
  else
  {
    while ( 1 )
    {
      if ( *(_BYTE *)(*v18 + 16) != 77 )
      {
        v21 = sub_19E1D90(*v18);
        if ( !v21 || *(_BYTE *)(v21 + 16) != 77 )
          break;
      }
      do
        ++v18;
      while ( v20 != v18 && (unsigned __int64)*v18 >= 0xFFFFFFFFFFFFFFFELL );
      if ( v18 == v19 )
        goto LABEL_37;
    }
    v39 = 2;
    LODWORD(a2) = 0;
  }
  v22 = *(_QWORD *)(v13 + 16);
  if ( v22 == *(_QWORD *)(v13 + 8) )
    v23 = v22 + 8LL * *(unsigned int *)(v13 + 28);
  else
    v23 = v22 + 8LL * *(unsigned int *)(v13 + 24);
  v43[0] = *(_QWORD *)(v13 + 16);
  v43[1] = v23;
  sub_19E4730((__int64)v43);
  v43[2] = v13;
  v43[3] = *(_QWORD *)v13;
  v24 = *(_QWORD *)(v13 + 16);
  if ( v24 == *(_QWORD *)(v13 + 8) )
    v25 = *(unsigned int *)(v13 + 28);
  else
    v25 = *(unsigned int *)(v13 + 24);
  v44[0] = v24 + 8 * v25;
  v44[1] = v44[0];
  sub_19E4730((__int64)v44);
  v44[2] = v13;
  v44[3] = *(_QWORD *)v13;
  v26 = (__int64 *)v43[0];
  if ( v43[0] != v44[0] )
  {
    while ( 1 )
    {
      v27 = *v26;
      if ( *(_BYTE *)(v27 + 16) == 77 )
      {
        v41 = v27;
        v42 = v39;
        v28 = sub_19E1070(a1 + 2024, &v41, &v40);
        v29 = v40;
        if ( !v28 )
          break;
      }
LABEL_26:
      v43[0] += 8LL;
      sub_19E4730((__int64)v43);
      v26 = (__int64 *)v43[0];
      if ( v43[0] == v44[0] )
        return (unsigned int)a2;
    }
    v30 = *(_DWORD *)(a1 + 2048);
    v31 = *(_DWORD *)(a1 + 2040);
    ++*(_QWORD *)(a1 + 2024);
    v32 = v31 + 1;
    if ( 4 * v32 >= 3 * v30 )
    {
      v30 *= 2;
    }
    else if ( v30 - *(_DWORD *)(a1 + 2044) - v32 > v30 >> 3 )
    {
LABEL_31:
      *(_DWORD *)(a1 + 2040) = v32;
      if ( *(_QWORD *)v29 != -8 )
        --*(_DWORD *)(a1 + 2044);
      *(_QWORD *)v29 = v41;
      *(_DWORD *)(v29 + 8) = v42;
      goto LABEL_26;
    }
    sub_19E3FB0(a1 + 2024, v30);
    sub_19E1070(a1 + 2024, &v41, &v40);
    v29 = v40;
    v32 = *(_DWORD *)(a1 + 2040) + 1;
    goto LABEL_31;
  }
  return (unsigned int)a2;
}
