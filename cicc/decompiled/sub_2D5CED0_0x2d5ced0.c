// Function: sub_2D5CED0
// Address: 0x2d5ced0
//
__int64 __fastcall sub_2D5CED0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // rbx
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r15
  __int64 v9; // r12
  __int64 v10; // rax
  char v11; // al
  bool v12; // zf
  unsigned int v13; // eax
  __int64 v14; // r12
  __int64 v15; // r12
  __int64 v16; // r13
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rcx
  _QWORD *v25; // rax
  _QWORD *v26; // r14
  __int64 v27; // r12
  __int64 v28; // r13
  __int64 v29; // r12
  __int64 v30; // r8
  unsigned __int64 v31; // r9
  __int64 v32; // rax
  _QWORD *v33; // rax
  char *v34; // rbx
  __int64 v35; // rdx
  unsigned __int64 v36; // rcx
  unsigned __int64 v37; // rsi
  int v38; // eax
  _QWORD *v39; // rdx
  __int64 result; // rax
  __int64 v41; // rdx
  char *v42; // rbx
  unsigned __int64 v43; // [rsp+8h] [rbp-A8h]
  __int64 v44; // [rsp+10h] [rbp-A0h]
  __int64 v45; // [rsp+10h] [rbp-A0h]
  __int64 v46; // [rsp+18h] [rbp-98h]
  unsigned int v47; // [rsp+18h] [rbp-98h]
  const void *v48; // [rsp+20h] [rbp-90h]
  const void *v49; // [rsp+20h] [rbp-90h]
  __int64 v50; // [rsp+28h] [rbp-88h]
  _QWORD v52[12]; // [rsp+50h] [rbp-60h] BYREF

  v4 = a2;
  v50 = *(_QWORD *)(a1 + 144);
  v5 = sub_22077B0(0x90u);
  v8 = v5;
  if ( !v5 )
  {
    v34 = (char *)v52;
    goto LABEL_38;
  }
  *(_QWORD *)(v5 + 8) = a2;
  v9 = *(_QWORD *)(a2 + 40);
  *(_QWORD *)v5 = off_49D4180;
  *(_QWORD *)(v5 + 16) = 0;
  *(_WORD *)(v5 + 24) = 0;
  *(_BYTE *)(v5 + 48) = 0;
  v10 = *(_QWORD *)(v9 + 56);
  if ( v10 )
  {
    v11 = a2 != v10 - 24;
    v12 = *(_BYTE *)(v9 + 40) == 0;
    *(_BYTE *)(v8 + 56) = v11;
    if ( v12 )
      goto LABEL_4;
  }
  else
  {
    v12 = *(_BYTE *)(v9 + 40) == 0;
    *(_BYTE *)(v8 + 56) = 1;
    if ( v12 )
      goto LABEL_44;
  }
  *(_QWORD *)(v8 + 40) = sub_B43FD0(a2);
  v11 = *(_BYTE *)(v8 + 56);
  *(_QWORD *)(v8 + 48) = v41;
LABEL_4:
  if ( v11 )
  {
LABEL_44:
    *(_QWORD *)(v8 + 16) = *(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    *(_WORD *)(v8 + 24) = 0;
    goto LABEL_6;
  }
  *(_QWORD *)(v8 + 32) = v9;
LABEL_6:
  *(_QWORD *)(v8 + 72) = a2;
  *(_QWORD *)(v8 + 64) = &off_49D4060;
  v48 = (const void *)(v8 + 96);
  *(_QWORD *)(v8 + 80) = v8 + 96;
  *(_QWORD *)(v8 + 88) = 0x400000000LL;
  v13 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v14 = v13;
  if ( v13 > 4uLL )
  {
    v47 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    sub_C8D5F0(v8 + 80, v48, v47, 8u, v6, v7);
    v13 = v47;
  }
  v15 = 32 * v14;
  if ( v13 )
  {
    v16 = 0;
    do
    {
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        v17 = *(_QWORD *)(a2 - 8);
      else
        v17 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      v18 = *(unsigned int *)(v8 + 88);
      v19 = *(_QWORD *)(v17 + v16);
      if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(v8 + 92) )
      {
        v45 = *(_QWORD *)(v17 + v16);
        sub_C8D5F0(v8 + 80, v48, v18 + 1, 8u, v6, v19);
        v18 = *(unsigned int *)(v8 + 88);
        v19 = v45;
      }
      *(_QWORD *)(*(_QWORD *)(v8 + 80) + 8 * v18) = v19;
      ++*(_DWORD *)(v8 + 88);
      v20 = sub_ACADE0(*(__int64 ***)(v19 + 8));
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        v21 = *(_QWORD *)(a2 - 8);
      else
        v21 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      v22 = v16 + v21;
      if ( *(_QWORD *)v22 )
      {
        v23 = *(_QWORD *)(v22 + 8);
        **(_QWORD **)(v22 + 16) = v23;
        if ( v23 )
          *(_QWORD *)(v23 + 16) = *(_QWORD *)(v22 + 16);
      }
      *(_QWORD *)v22 = v20;
      if ( v20 )
      {
        v24 = *(_QWORD *)(v20 + 16);
        *(_QWORD *)(v22 + 8) = v24;
        if ( v24 )
          *(_QWORD *)(v24 + 16) = v22 + 8;
        *(_QWORD *)(v22 + 16) = v20 + 16;
        *(_QWORD *)(v20 + 16) = v22;
      }
      v16 += 32;
    }
    while ( v16 != v15 );
    v4 = a2;
  }
  *(_QWORD *)(v8 + 128) = 0;
  *(_QWORD *)(v8 + 136) = v50;
  if ( a3 )
  {
    v25 = (_QWORD *)sub_22077B0(0x98u);
    v26 = v25;
    if ( v25 )
    {
      v25[1] = v4;
      v27 = *(_QWORD *)(v4 + 16);
      *v25 = off_49D4150;
      v49 = v25 + 4;
      v25[2] = v25 + 4;
      v25[3] = 0x400000000LL;
      v25[12] = v25 + 14;
      v25[13] = 0x100000000LL;
      v25[16] = 0x100000000LL;
      v25[15] = v25 + 17;
      v25[18] = a3;
      v46 = (__int64)(v25 + 2);
      if ( v27 )
      {
        v44 = v4;
        v28 = v27;
        do
        {
          v29 = *(_QWORD *)(v28 + 24);
          v31 = (unsigned int)sub_BD2910(v28) | v3 & 0xFFFFFFFF00000000LL;
          v32 = *((unsigned int *)v26 + 6);
          v3 = v31;
          if ( v32 + 1 > (unsigned __int64)*((unsigned int *)v26 + 7) )
          {
            v43 = v31;
            sub_C8D5F0(v46, v49, v32 + 1, 0x10u, v30, v31);
            v32 = *((unsigned int *)v26 + 6);
            v31 = v43;
          }
          v33 = (_QWORD *)(v26[2] + 16 * v32);
          *v33 = v29;
          v33[1] = v31;
          ++*((_DWORD *)v26 + 6);
          v28 = *(_QWORD *)(v28 + 8);
        }
        while ( v28 );
        v4 = v44;
      }
      sub_AE7A40((__int64)(v26 + 12), (_BYTE *)v4, (__int64)(v26 + 15));
      sub_BD84D0(v4, a3);
    }
    *(_QWORD *)(v8 + 128) = v26;
  }
  v34 = (char *)v52;
  sub_BED950((__int64)v52, v50, v4);
  sub_B43D10((_QWORD *)v4);
LABEL_38:
  v52[0] = v8;
  v35 = *(unsigned int *)(a1 + 8);
  v36 = *(_QWORD *)a1;
  v37 = v35 + 1;
  v38 = *(_DWORD *)(a1 + 8);
  if ( v35 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    if ( v36 > (unsigned __int64)v52 || (unsigned __int64)v52 >= v36 + 8 * v35 )
    {
      sub_2D57B00(a1, v37, v35, v36, v6, v7);
      v35 = *(unsigned int *)(a1 + 8);
      v36 = *(_QWORD *)a1;
      v38 = *(_DWORD *)(a1 + 8);
    }
    else
    {
      v42 = (char *)v52 - v36;
      sub_2D57B00(a1, v37, v35, v36, v6, v7);
      v36 = *(_QWORD *)a1;
      v35 = *(unsigned int *)(a1 + 8);
      v34 = &v42[*(_QWORD *)a1];
      v38 = *(_DWORD *)(a1 + 8);
    }
  }
  v39 = (_QWORD *)(v36 + 8 * v35);
  if ( v39 )
  {
    *v39 = *(_QWORD *)v34;
    *(_QWORD *)v34 = 0;
    v8 = v52[0];
    v38 = *(_DWORD *)(a1 + 8);
  }
  result = (unsigned int)(v38 + 1);
  *(_DWORD *)(a1 + 8) = result;
  if ( v8 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v8 + 8LL))(v8);
  return result;
}
