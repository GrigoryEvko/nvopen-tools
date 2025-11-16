// Function: sub_2DAF430
// Address: 0x2daf430
//
__int64 __fastcall sub_2DAF430(__int64 *a1, __int64 a2)
{
  _BYTE *v2; // rax
  char v3; // r15
  __int64 v4; // rax
  __int64 *v5; // r9
  __int64 v6; // r10
  bool v7; // r14
  __int64 v8; // rbx
  __int64 v9; // r13
  __int64 v10; // r8
  __int64 v11; // r12
  int v12; // eax
  unsigned __int8 v13; // dl
  __int64 v14; // rax
  unsigned __int16 v15; // dx
  unsigned __int64 v16; // rdi
  unsigned __int64 *v17; // rbx
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // rdi
  __int64 v21; // rsi
  unsigned __int16 v22; // cx
  int v23; // ecx
  __int64 v24; // r11
  __int128 v25; // rax
  bool v26; // al
  unsigned __int8 v28; // [rsp+16h] [rbp-16Ah]
  bool v29; // [rsp+17h] [rbp-169h]
  __int64 *v30; // [rsp+20h] [rbp-160h]
  __int64 v31; // [rsp+28h] [rbp-158h]
  __int64 v32; // [rsp+30h] [rbp-150h]
  __int64 v33; // [rsp+38h] [rbp-148h]
  __int64 *v34; // [rsp+40h] [rbp-140h]
  __int64 v35; // [rsp+48h] [rbp-138h]
  __int64 v36[2]; // [rsp+50h] [rbp-130h] BYREF
  unsigned __int64 v37; // [rsp+60h] [rbp-120h]
  unsigned __int64 v38; // [rsp+68h] [rbp-118h]
  unsigned __int64 v39; // [rsp+90h] [rbp-F0h]
  __int64 v40; // [rsp+B0h] [rbp-D0h]
  char *v41; // [rsp+B8h] [rbp-C8h]
  char v42; // [rsp+C8h] [rbp-B8h] BYREF
  char *v43; // [rsp+100h] [rbp-80h]
  char v44; // [rsp+110h] [rbp-70h] BYREF

  v2 = *(_BYTE **)(a2 + 32);
  *a1 = (__int64)v2;
  v3 = v2[48];
  v28 = 0;
  if ( !v3 )
    return v28;
  v4 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)v2 + 16LL) + 200LL))(*(_QWORD *)(*(_QWORD *)v2 + 16LL));
  a1[1] = v4;
  sub_2DAE000((__int64)v36, *a1, v4);
  v5 = a1;
  v35 = a2 + 320;
  do
  {
    v34 = v5;
    sub_2DAF240(v36);
    v6 = *(_QWORD *)(a2 + 328);
    if ( v6 == v35 )
      break;
    v29 = 0;
    v5 = v34;
    v7 = 0;
    do
    {
      v8 = *(_QWORD *)(v6 + 56);
      v9 = v6 + 48;
      if ( v8 != v6 + 48 )
      {
        while ( 1 )
        {
          v10 = *(_QWORD *)(v8 + 32);
          v11 = v10 + 40LL * (*(_DWORD *)(v8 + 40) & 0xFFFFFF);
          if ( v10 != v11 )
            break;
LABEL_18:
          if ( (*(_BYTE *)v8 & 4) == 0 )
          {
            while ( (*(_BYTE *)(v8 + 44) & 8) != 0 )
              v8 = *(_QWORD *)(v8 + 8);
          }
          v8 = *(_QWORD *)(v8 + 8);
          if ( v9 == v8 )
            goto LABEL_20;
        }
        while ( 1 )
        {
          if ( *(_BYTE *)v10 )
            goto LABEL_17;
          v12 = *(_DWORD *)(v10 + 8);
          if ( v12 >= 0 )
            goto LABEL_17;
          v13 = *(_BYTE *)(v10 + 3);
          v14 = v37 + 32LL * (v12 & 0x7FFFFFFF);
          if ( (v13 & 0x10) != 0 && (((v13 & 0x10) != 0) & (v13 >> 6)) == 0 && !*(_QWORD *)v14 && !*(_QWORD *)(v14 + 8) )
          {
            v7 = v3;
            *(_BYTE *)(v10 + 3) = v13 | 0x40;
          }
          if ( (*(_BYTE *)(v10 + 4) & 1) != 0 )
            goto LABEL_17;
          if ( (*(_BYTE *)(v10 + 4) & 2) != 0 )
            goto LABEL_17;
          v15 = (*(_DWORD *)v10 >> 8) & 0xFFF;
          if ( (*(_BYTE *)(v10 + 3) & 0x10) != 0 && !v15 )
            goto LABEL_17;
          if ( (*(_OWORD *)(*(_QWORD *)(v5[1] + 272) + 16LL * v15) & *(_OWORD *)(v14 + 16) & *(_OWORD *)v14) == 0 )
            goto LABEL_16;
          if ( (*(_BYTE *)(v10 + 3) & 0x10) != 0 )
            goto LABEL_17;
          v21 = *(_QWORD *)(v10 + 16);
          v22 = *(_WORD *)(v21 + 68);
          if ( v22 > 0x14u )
            goto LABEL_17;
          if ( ((1LL << v22) & 0x180301) == 0 )
            goto LABEL_17;
          v23 = *(_DWORD *)(*(_QWORD *)(v21 + 32) + 8LL);
          if ( v23 >= 0 )
            goto LABEL_17;
          v24 = v23 & 0x7FFFFFFF;
          if ( (*(_QWORD *)&v43[8 * ((unsigned int)v24 >> 6)] & (1LL << v23)) == 0 )
            goto LABEL_17;
          v30 = v5;
          v31 = v6;
          v33 = v10;
          v32 = v23 & 0x7FFFFFFF;
          *(_QWORD *)&v25 = sub_2DAE370(v36, v21, *(_QWORD *)(v37 + 32 * v24), *(_QWORD *)(v37 + 32 * v24 + 8), v10);
          v10 = v33;
          v6 = v31;
          v5 = v30;
          if ( v25 != 0 )
            goto LABEL_17;
          if ( *(int *)(v33 + 8) >= 0 )
          {
LABEL_16:
            *(_BYTE *)(v10 + 4) |= 1u;
            v7 = v3;
          }
          else
          {
            v26 = sub_2DADC20(
                    (_QWORD *)*v30,
                    v21,
                    *(_QWORD *)(*(_QWORD *)(*v30 + 56) + 16 * v32) & 0xFFFFFFFFFFFFFFF8LL,
                    (_DWORD *)v33);
            v10 = v33;
            v5 = v30;
            v6 = v31;
            v7 = v26;
            *(_BYTE *)(v33 + 4) |= 1u;
            if ( v26 )
              v29 = v26;
            else
              v7 = v3;
          }
LABEL_17:
          v10 += 40;
          if ( v11 == v10 )
            goto LABEL_18;
        }
      }
LABEL_20:
      v6 = *(_QWORD *)(v6 + 8);
    }
    while ( v6 != v35 );
    v28 |= v7;
  }
  while ( v29 );
  if ( v43 != &v44 )
    _libc_free((unsigned __int64)v43);
  if ( v41 != &v42 )
    _libc_free((unsigned __int64)v41);
  v16 = v38;
  if ( v38 )
  {
    v17 = (unsigned __int64 *)v39;
    v18 = v40 + 8;
    if ( v40 + 8 > v39 )
    {
      do
      {
        v19 = *v17++;
        j_j___libc_free_0(v19);
      }
      while ( v18 > (unsigned __int64)v17 );
      v16 = v38;
    }
    j_j___libc_free_0(v16);
  }
  if ( v37 )
    j_j___libc_free_0_0(v37);
  return v28;
}
