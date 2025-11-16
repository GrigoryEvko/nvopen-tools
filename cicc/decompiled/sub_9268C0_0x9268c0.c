// Function: sub_9268C0
// Address: 0x9268c0
//
__int64 __fastcall sub_9268C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // r15
  __int64 v8; // r10
  unsigned int v9; // r8d
  char v10; // al
  __int64 v11; // rax
  char v12; // dl
  char v13; // bl
  unsigned int v14; // eax
  int v15; // eax
  __int64 *v17; // r9
  unsigned __int8 *v18; // rax
  int v19; // edx
  __int64 v20; // rcx
  __int64 v21; // r13
  unsigned __int64 v22; // rsi
  __int64 v23; // rax
  __int64 *v24; // rdi
  __int64 v25; // rax
  unsigned __int64 v26; // rsi
  int v27; // ecx
  unsigned __int64 v28; // rax
  __int64 v29; // rax
  unsigned __int8 v30; // al
  int v31; // r8d
  __int64 v32; // rax
  int v33; // r9d
  __int64 v34; // rbx
  unsigned int *v35; // r15
  unsigned int *v36; // r14
  __int64 v37; // rdx
  __int64 v38; // rsi
  char v39; // al
  int v40; // ebx
  _BOOL4 v41; // edx
  int v42; // [rsp+0h] [rbp-A0h]
  int v43; // [rsp+4h] [rbp-9Ch]
  int v44; // [rsp+4h] [rbp-9Ch]
  int v45; // [rsp+4h] [rbp-9Ch]
  int v46; // [rsp+8h] [rbp-98h]
  __int64 v47; // [rsp+8h] [rbp-98h]
  unsigned int v48; // [rsp+10h] [rbp-90h]
  unsigned int v49; // [rsp+10h] [rbp-90h]
  __int64 v50; // [rsp+18h] [rbp-88h]
  unsigned __int64 v51; // [rsp+18h] [rbp-88h]
  unsigned int v52; // [rsp+2Ch] [rbp-74h] BYREF
  const char *v53; // [rsp+30h] [rbp-70h] BYREF
  __int64 v54; // [rsp+38h] [rbp-68h]
  unsigned int v55; // [rsp+48h] [rbp-58h]
  __int16 v56; // [rsp+50h] [rbp-50h]

  v5 = *(_QWORD *)(a3 + 72);
  v6 = *(_QWORD *)(v5 + 16);
  if ( *(_BYTE *)(v5 + 24) != 3 )
    goto LABEL_2;
  if ( !(unsigned __int8)sub_920430(*(_QWORD *)(a2 + 32), *(_QWORD *)(v5 + 56), &v52) )
    goto LABEL_2;
  if ( v52 > 3 )
    goto LABEL_2;
  if ( *(_BYTE *)(v6 + 24) != 4 )
    goto LABEL_2;
  v18 = *(unsigned __int8 **)(*(_QWORD *)(v6 + 56) + 8LL);
  if ( !v18 )
    goto LABEL_2;
  v19 = *v18;
  if ( v19 == 120 )
  {
    v20 = v18[1];
    if ( !v18[1] )
      goto LABEL_17;
  }
  if ( v19 == 121 )
  {
    v20 = 1;
    if ( !v18[1] )
      goto LABEL_17;
  }
  if ( v19 != 122 || v18[1] )
  {
LABEL_2:
    sub_926800((__int64)&v53, a2, v5);
    v7 = *(_QWORD *)v5;
    v8 = v54;
    v9 = v55;
    v10 = *(_BYTE *)(*(_QWORD *)v5 + 140LL);
    if ( v10 == 12 )
    {
      v11 = *(_QWORD *)v5;
      do
      {
        v11 = *(_QWORD *)(v11 + 160);
        v12 = *(_BYTE *)(v11 + 140);
      }
      while ( v12 == 12 );
      v13 = v12 == 11;
    }
    else
    {
      v13 = v10 == 11;
      if ( (v10 & 0xFB) != 8 )
      {
        LOBYTE(v15) = 0;
        goto LABEL_9;
      }
    }
    v48 = v55;
    v50 = v54;
    v14 = sub_8D4C10(v7, dword_4F077C4 != 2);
    v8 = v50;
    v9 = v48;
    v15 = (v14 >> 1) & 1;
LABEL_9:
    sub_9229B0(a1, a2, v8, v7, v9, v6, v13, v15);
    return a1;
  }
  v20 = 2;
LABEL_17:
  v21 = *v17;
  v46 = v20;
  v49 = v52;
  v22 = *v17;
  v53 = "predef_tmp_comp";
  v56 = 259;
  v23 = sub_921CE0(a2, v22, (__int64)&v53, v20);
  v24 = *(__int64 **)(a2 + 32);
  v51 = v23;
  v56 = 257;
  v25 = sub_90A810(v24, dword_3F109E0[3 * v49 + v46], 0, 0);
  v26 = 0;
  if ( v25 )
    v26 = *(_QWORD *)(v25 + 24);
  v47 = sub_921880((unsigned int **)(a2 + 48), v26, v25, 0, 0, (__int64)&v53, 0);
  v27 = unk_4D0463C;
  if ( unk_4D0463C )
    v27 = sub_90AA40(*(_QWORD *)(a2 + 32), v51);
  if ( *(char *)(v21 + 142) >= 0 && *(_BYTE *)(v21 + 140) == 12 )
  {
    v45 = v27;
    LODWORD(v28) = sub_8D4AB0(v21);
    v27 = v45;
    v28 = (unsigned int)v28;
  }
  else
  {
    v28 = *(unsigned int *)(v21 + 136);
  }
  if ( v28 )
  {
    _BitScanReverse64(&v28, v28);
    v31 = (unsigned __int8)(63 - (v28 ^ 0x3F));
  }
  else
  {
    v43 = v27;
    v29 = sub_AA4E30(*(_QWORD *)(a2 + 96));
    v30 = sub_AE5020(v29, *(_QWORD *)(v47 + 8));
    v27 = v43;
    v31 = v30;
  }
  v42 = v31;
  v56 = 257;
  v44 = v27;
  v32 = sub_BD2C40(80, unk_3F10A10);
  v34 = v32;
  if ( v32 )
    sub_B4D3C0(v32, v47, v51, v44, v42, v33, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, const char **, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
    *(_QWORD *)(a2 + 136),
    v34,
    &v53,
    *(_QWORD *)(a2 + 104),
    *(_QWORD *)(a2 + 112));
  v35 = *(unsigned int **)(a2 + 48);
  v36 = &v35[4 * *(unsigned int *)(a2 + 56)];
  while ( v36 != v35 )
  {
    v37 = *((_QWORD *)v35 + 1);
    v38 = *v35;
    v35 += 4;
    sub_B99FD0(v34, v38, v37);
  }
  v39 = *(_BYTE *)(v21 + 140);
  if ( *(char *)(v21 + 142) >= 0 && v39 == 12 )
  {
    v40 = sub_8D4AB0(v21);
    v39 = *(_BYTE *)(v21 + 140);
  }
  else
  {
    v40 = *(_DWORD *)(v21 + 136);
  }
  v41 = 0;
  if ( (v39 & 0xFB) == 8 )
    v41 = (sub_8D4C10(v21, dword_4F077C4 != 2) & 2) != 0;
  *(_DWORD *)a1 = 0;
  *(_QWORD *)(a1 + 16) = v21;
  *(_QWORD *)(a1 + 8) = v51;
  *(_DWORD *)(a1 + 48) = v41;
  *(_DWORD *)(a1 + 24) = v40;
  return a1;
}
