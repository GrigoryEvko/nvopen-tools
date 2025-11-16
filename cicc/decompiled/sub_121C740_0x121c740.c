// Function: sub_121C740
// Address: 0x121c740
//
__int64 __fastcall sub_121C740(__int64 a1, unsigned int a2, __int64 a3, unsigned __int64 a4)
{
  int v6; // eax
  __int64 v9; // rsi
  int v10; // ecx
  unsigned int v11; // edx
  int *v12; // rax
  int v13; // edi
  __int64 v14; // r8
  _QWORD *v15; // r13
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 v19; // rsi
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // r8
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  _BOOL8 v29; // rdi
  int v30; // eax
  int v31; // r8d
  __int64 v32; // rdi
  _QWORD **v33; // [rsp+0h] [rbp-80h]
  __int64 v34; // [rsp+0h] [rbp-80h]
  _QWORD *v35; // [rsp+8h] [rbp-78h]
  __int64 v36; // [rsp+8h] [rbp-78h]
  __int64 v37; // [rsp+8h] [rbp-78h]
  __int64 v38; // [rsp+18h] [rbp-68h]
  _QWORD v39[2]; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v40; // [rsp+30h] [rbp-50h]
  __int16 v41; // [rsp+40h] [rbp-40h]

  if ( *(_BYTE *)(a3 + 8) != 14 )
  {
    v41 = 259;
    v15 = 0;
    v39[0] = "global variable reference must have pointer type";
    sub_11FD800(a1 + 176, a4, (__int64)v39, 1);
    return (__int64)v15;
  }
  v6 = *(_DWORD *)(a1 + 1216);
  v9 = *(_QWORD *)(a1 + 1200);
  if ( v6 )
  {
    v10 = v6 - 1;
    v11 = (v6 - 1) & (37 * a2);
    v12 = (int *)(v9 + 16LL * v11);
    v13 = *v12;
    if ( a2 == *v12 )
    {
LABEL_4:
      v14 = *((_QWORD *)v12 + 1);
      if ( v14 )
      {
LABEL_5:
        v41 = 2307;
        v39[0] = "@";
        v40 = a2;
        return sub_120A960(a1, a4, (__int64)v39, a3, v14);
      }
    }
    else
    {
      v30 = 1;
      while ( v13 != -1 )
      {
        v31 = v30 + 1;
        v11 = v10 & (v30 + v11);
        v12 = (int *)(v9 + 16LL * v11);
        v13 = *v12;
        if ( *v12 == a2 )
          goto LABEL_4;
        v30 = v31;
      }
    }
  }
  v17 = *(_QWORD *)(a1 + 1160);
  v18 = a1 + 1152;
  if ( v17 )
  {
    v19 = a1 + 1152;
    do
    {
      while ( 1 )
      {
        v20 = *(_QWORD *)(v17 + 16);
        v21 = *(_QWORD *)(v17 + 24);
        if ( *(_DWORD *)(v17 + 32) >= a2 )
          break;
        v17 = *(_QWORD *)(v17 + 24);
        if ( !v21 )
          goto LABEL_12;
      }
      v19 = v17;
      v17 = *(_QWORD *)(v17 + 16);
    }
    while ( v20 );
LABEL_12:
    if ( v18 != v19 && *(_DWORD *)(v19 + 32) <= a2 )
    {
      v14 = *(_QWORD *)(v19 + 40);
      if ( v14 )
        goto LABEL_5;
    }
  }
  v33 = *(_QWORD ***)(a1 + 344);
  BYTE4(v38) = 1;
  v35 = (_QWORD *)sub_BCB2B0(*v33);
  v41 = 257;
  LODWORD(v38) = *(_DWORD *)(a3 + 8) >> 8;
  v15 = sub_BD2C40(88, unk_3F0FAE8);
  if ( v15 )
    sub_B30000((__int64)v15, (__int64)v33, v35, 0, 9, 0, (__int64)v39, 0, 0, v38, 0);
  v22 = *(_QWORD *)(a1 + 1160);
  v23 = a1 + 1152;
  if ( !v22 )
    goto LABEL_23;
  do
  {
    while ( 1 )
    {
      v24 = *(_QWORD *)(v22 + 16);
      v25 = *(_QWORD *)(v22 + 24);
      if ( *(_DWORD *)(v22 + 32) >= a2 )
        break;
      v22 = *(_QWORD *)(v22 + 24);
      if ( !v25 )
        goto LABEL_21;
    }
    v23 = v22;
    v22 = *(_QWORD *)(v22 + 16);
  }
  while ( v24 );
LABEL_21:
  if ( v18 == v23 || *(_DWORD *)(v23 + 32) > a2 )
  {
LABEL_23:
    v34 = v23;
    v26 = sub_22077B0(56);
    *(_DWORD *)(v26 + 32) = a2;
    *(_QWORD *)(v26 + 40) = 0;
    *(_QWORD *)(v26 + 48) = 0;
    v36 = v26;
    v27 = sub_121C640((_QWORD *)(a1 + 1144), v34, (unsigned int *)(v26 + 32));
    if ( v28 )
    {
      v29 = v27 || v18 == v28 || a2 < *(_DWORD *)(v28 + 32);
      sub_220F040(v29, v36, v28, a1 + 1152);
      v23 = v36;
      ++*(_QWORD *)(a1 + 1184);
    }
    else
    {
      v32 = v36;
      v37 = v27;
      j_j___libc_free_0(v32, 56);
      v23 = v37;
    }
  }
  *(_QWORD *)(v23 + 40) = v15;
  *(_QWORD *)(v23 + 48) = a4;
  return (__int64)v15;
}
