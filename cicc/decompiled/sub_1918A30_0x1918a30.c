// Function: sub_1918A30
// Address: 0x1918a30
//
__int64 __fastcall sub_1918A30(__int64 a1, __int64 a2)
{
  __int64 v4; // r13
  unsigned __int8 v5; // al
  unsigned int v6; // r14d
  unsigned int v7; // r15d
  __int64 *v9; // rax
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // r13
  __int64 v13; // rdx
  __int64 v14; // rcx
  unsigned __int64 v15; // rdi
  __int64 v16; // r8
  __int64 v17; // r9
  unsigned int v18; // r14d
  unsigned int v19; // esi
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r12
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rbx
  int v28; // eax
  __int64 v29; // rax
  __int64 v30; // rbx
  _QWORD *v31; // rax
  __int64 **v32; // r13
  __int64 v33; // r14
  __int64 **v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r13
  _QWORD *v38; // rdi
  int v39; // r8d
  int v40; // r9d
  __int64 v41; // rax
  int v42; // [rsp+14h] [rbp-5Ch]
  unsigned __int64 v43; // [rsp+18h] [rbp-58h]
  __int64 v44; // [rsp+28h] [rbp-48h] BYREF
  __int64 v45[8]; // [rsp+30h] [rbp-40h] BYREF

  v4 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v5 = *(_BYTE *)(v4 + 16);
  v44 = v4;
  if ( v5 == 13 )
  {
    v6 = *(_DWORD *)(v4 + 32);
    if ( v6 <= 0x40 )
    {
      if ( *(_QWORD *)(v4 + 24) )
        goto LABEL_4;
    }
    else if ( v6 != (unsigned int)sub_16A57B0(v4 + 24) )
    {
      goto LABEL_4;
    }
    v31 = (_QWORD *)sub_16498A0(v4);
    v32 = (__int64 **)sub_1643330(v31);
    v33 = sub_1599EF0(v32);
    v34 = (__int64 **)sub_1647190((__int64 *)v32, 0);
    v37 = sub_15A06D0(v34, 0, v35, v36);
    v38 = sub_1648A60(64, 2u);
    if ( v38 )
    {
      sub_15F9660((__int64)v38, v33, v37, a2);
      if ( !sub_1602380(a2) )
        return 0;
      goto LABEL_23;
    }
LABEL_4:
    if ( !sub_1602380(a2) )
      return 0;
LABEL_23:
    sub_190ACD0(a1 + 152, a2);
    v41 = *(unsigned int *)(a1 + 680);
    if ( (unsigned int)v41 >= *(_DWORD *)(a1 + 684) )
    {
      sub_16CD150(a1 + 672, (const void *)(a1 + 688), 0, 8, v39, v40);
      v41 = *(unsigned int *)(a1 + 680);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 672) + 8 * v41) = a2;
    ++*(_DWORD *)(a1 + 680);
    return 0;
  }
  if ( v5 <= 0x10u )
    return 0;
  v9 = (__int64 *)sub_16498A0(v4);
  v10 = sub_159C4F0(v9);
  v11 = *(_QWORD *)(a2 + 40);
  v12 = v10;
  v15 = sub_157EBA0(v11);
  if ( v15 && (v42 = sub_15F4D60(v15), v43 = sub_157EBA0(v11), v42) )
  {
    v18 = 0;
    v7 = 0;
    do
    {
      v19 = v18++;
      v20 = sub_15F4DF0(v43, v19);
      v21 = *(_QWORD *)(a2 + 40);
      v45[1] = v20;
      v45[0] = v21;
      v7 |= sub_19166D0(a1, v44, v12, v45, 0);
    }
    while ( v42 != v18 );
  }
  else
  {
    v7 = 0;
  }
  v22 = a1 + 512;
  *(_QWORD *)sub_1918850(v22, &v44, v13, v14, v16, v17) = v12;
  v27 = v44;
  if ( (unsigned __int8)(*(_BYTE *)(v44 + 16) - 75) <= 1u )
  {
    v28 = *(unsigned __int16 *)(v44 + 18);
    BYTE1(v28) &= ~0x80u;
    if ( v28 == 1 || v28 == 32 || v28 == 9 && (sub_15F24E0(v44) & 2) != 0 )
    {
      v29 = *(_QWORD *)(v27 - 48);
      v45[0] = v29;
      v30 = *(_QWORD *)(v27 - 24);
      if ( *(_BYTE *)(v29 + 16) > 0x10u )
      {
        if ( *(_BYTE *)(v30 + 16) > 0x10u )
          return v7;
      }
      else
      {
        v45[0] = v30;
        if ( *(_BYTE *)(v30 + 16) <= 0x10u )
          return v7;
        v30 = v29;
      }
      *(_QWORD *)sub_1918850(v22, v45, v23, v24, v25, v26) = v30;
    }
  }
  return v7;
}
