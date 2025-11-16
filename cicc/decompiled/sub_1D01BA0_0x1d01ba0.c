// Function: sub_1D01BA0
// Address: 0x1d01ba0
//
void __fastcall sub_1D01BA0(__int64 a1, __int64 *a2)
{
  int v2; // eax
  _QWORD *v3; // r12
  _QWORD *v4; // rbx
  _DWORD *v5; // rax
  __int64 v6; // r15
  int v7; // eax
  int v8; // esi
  __int64 v9; // rsi
  int v10; // r14d
  __int64 v11; // r13
  _QWORD *v12; // rbx
  __int64 v13; // r12
  __int64 v14; // r15
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 (__fastcall *v18)(__int64, unsigned __int8); // rcx
  __int64 v19; // rcx
  __int64 (__fastcall *v20)(__int64, unsigned __int8); // rax
  __int64 v21; // rcx
  unsigned int v22; // r10d
  __int64 v23; // r9
  unsigned __int8 v24; // al
  __int64 v25; // rdi
  __int64 (__fastcall *v26)(__int64, unsigned __int8); // rax
  unsigned __int8 v27; // al
  int v28; // ebx
  unsigned int i; // r14d
  __int64 v30; // r13
  __int64 v31; // rdi
  __int64 v32; // rax
  __int64 (__fastcall *v33)(__int64, unsigned __int8); // rdx
  __int64 *v34; // rdx
  __int64 v35; // rdx
  __int64 (__fastcall *v36)(__int64, unsigned __int8); // rax
  unsigned __int16 v37; // cx
  int v38; // edx
  int v39; // eax
  __int64 v40; // rdi
  __int64 v41; // r13
  __int64 v42; // rax
  __int64 (__fastcall *v43)(__int64, unsigned __int8); // rdx
  __int64 v44; // rdx
  __int64 (__fastcall *v45)(__int64, unsigned __int8); // rax
  unsigned __int16 v46; // r14
  int v47; // edx
  __int64 v48; // rax
  __int64 v49; // rax
  unsigned __int8 v50; // al
  __int64 v51; // rax
  _QWORD *v52; // [rsp+8h] [rbp-68h]
  __int64 v53; // [rsp+10h] [rbp-60h]
  unsigned int v54; // [rsp+1Ch] [rbp-54h]
  __int64 v55; // [rsp+20h] [rbp-50h]
  __int64 v56; // [rsp+20h] [rbp-50h]
  __int64 v57; // [rsp+28h] [rbp-48h]
  _QWORD *v59; // [rsp+38h] [rbp-38h]
  unsigned __int16 v60; // [rsp+38h] [rbp-38h]

  if ( !*(_BYTE *)(a1 + 44) )
    return;
  v57 = *a2;
  if ( !*a2 )
    return;
  v2 = *(__int16 *)(*a2 + 24);
  v3 = (_QWORD *)a1;
  if ( (v2 & 0x8000u) != 0 )
  {
    v39 = ~v2;
    if ( (unsigned int)(v39 - 7) <= 3 || v39 == 14 )
      return;
  }
  else if ( (_WORD)v2 != 46 )
  {
    return;
  }
  v4 = (_QWORD *)a2[4];
  v59 = &v4[2 * *((unsigned int *)a2 + 10)];
  if ( v4 == v59 )
    goto LABEL_26;
  while ( 2 )
  {
    while ( (*v4 & 6) == 0 )
    {
      v5 = (_DWORD *)(*v4 & 0xFFFFFFFFFFFFFFF8LL);
      if ( v5[53] != v5[30] )
        break;
      v6 = *(_QWORD *)v5;
      v7 = *(__int16 *)(*(_QWORD *)v5 + 24LL);
      if ( (v7 & 0x8000u) == 0 )
      {
        if ( (_WORD)v7 != 47 )
          break;
LABEL_43:
        v40 = v3[10];
        v41 = **(unsigned __int8 **)(v6 + 40);
        v42 = *(_QWORD *)v40;
        v43 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v40 + 296LL);
        if ( v43 == sub_1D00B40 )
        {
          v44 = *(_QWORD *)(v40 + 8LL * (unsigned __int8)v41 + 1272);
        }
        else
        {
          v49 = v43(v40, **(_BYTE **)(v6 + 40));
          v40 = v3[10];
          v44 = v49;
          v42 = *(_QWORD *)v40;
        }
        v45 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(v42 + 304);
        v46 = *(_WORD *)(*(_QWORD *)v44 + 24LL);
        if ( v45 == sub_1D00B50 )
          v47 = *(unsigned __int8 *)(v40 + v41 + 2192);
        else
          v47 = (unsigned __int8)v45(v40, v41);
        *(_DWORD *)(v3[15] + 4LL * v46) += v47;
        break;
      }
      v8 = ~v7;
      if ( v7 == -10 )
        break;
      if ( (unsigned int)(-v7 - 8) <= 1 || v8 == 10 )
        goto LABEL_43;
      v9 = *(_QWORD *)(v3[8] + 8LL) + ((__int64)v8 << 6);
      v10 = *(unsigned __int8 *)(v9 + 4);
      if ( !*(_BYTE *)(v9 + 4) )
        break;
      v52 = v4;
      v11 = 0;
      v12 = v3;
      v13 = v6;
      do
      {
        while ( 1 )
        {
          v14 = *(unsigned __int8 *)(*(_QWORD *)(v13 + 40) + 16 * v11);
          if ( !(unsigned __int8)sub_1D18C40(v13) )
            goto LABEL_15;
          v16 = v12[10];
          v17 = *(_QWORD *)v16;
          v18 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v16 + 296LL);
          if ( v18 == sub_1D00B40 )
          {
            v19 = *(_QWORD *)(v16 + 8LL * (unsigned __int8)v14 + 1272);
          }
          else
          {
            v48 = v18(v16, v14);
            v16 = v12[10];
            v19 = v48;
            v17 = *(_QWORD *)v16;
          }
          v20 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(v17 + 304);
          v21 = *(unsigned __int16 *)(*(_QWORD *)v19 + 24LL);
          v22 = *(_DWORD *)(v12[15] + 4 * v21);
          v23 = 4 * v21;
          if ( v20 == sub_1D00B50 )
          {
            v24 = *(_BYTE *)(v16 + (unsigned __int8)v14 + 2192);
          }
          else
          {
            v53 = v21;
            v54 = *(_DWORD *)(v12[15] + 4 * v21);
            v55 = 4 * v21;
            v24 = v20(v16, v14);
            v21 = v53;
            v22 = v54;
            v23 = v55;
          }
          if ( v22 >= v24 )
            break;
          *(_DWORD *)(v12[15] + 4 * v21) = 0;
LABEL_15:
          if ( v10 == (_DWORD)++v11 )
            goto LABEL_25;
        }
        v25 = v12[10];
        v26 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v25 + 304LL);
        if ( v26 == sub_1D00B50 )
        {
          v27 = *(_BYTE *)(v25 + v14 + 2192);
        }
        else
        {
          v56 = v23;
          v27 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64, __int64))v26)(v25, (unsigned int)v14, v15, v21);
          v23 = v56;
        }
        ++v11;
        *(_DWORD *)(v12[15] + v23) -= v27;
      }
      while ( v10 != (_DWORD)v11 );
LABEL_25:
      v3 = v12;
      v4 = v52 + 2;
      if ( v59 == v52 + 2 )
      {
LABEL_26:
        if ( *((_DWORD *)a2 + 51) )
          goto LABEL_27;
        return;
      }
    }
    v4 += 2;
    if ( v59 != v4 )
      continue;
    break;
  }
  if ( *((_DWORD *)a2 + 51) )
  {
LABEL_27:
    if ( *(__int16 *)(v57 + 24) < 0 )
    {
      v28 = *(_DWORD *)(v57 + 60);
      for ( i = *(unsigned __int8 *)(*(_QWORD *)(v3[8] + 8LL) + ((__int64)~*(__int16 *)(v57 + 24) << 6) + 4); v28 != i; ++i )
      {
        v30 = *(unsigned __int8 *)(*(_QWORD *)(v57 + 40) + 16LL * i);
        if ( (_BYTE)v30 != 1 && (_BYTE)v30 != 111 && (unsigned __int8)sub_1D18C40(v57) )
        {
          v31 = v3[10];
          v32 = *(_QWORD *)v31;
          v33 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v31 + 296LL);
          if ( v33 == sub_1D00B40 )
          {
            v34 = *(__int64 **)(v31 + 8LL * (unsigned __int8)v30 + 1272);
          }
          else
          {
            v51 = v33(v31, v30);
            v31 = v3[10];
            v34 = (__int64 *)v51;
            v32 = *(_QWORD *)v31;
          }
          v35 = *v34;
          v36 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(v32 + 304);
          v37 = *(_WORD *)(v35 + 24);
          if ( v36 == sub_1D00B50 )
          {
            v38 = *(unsigned __int8 *)(v31 + v30 + 2192);
          }
          else
          {
            v60 = *(_WORD *)(v35 + 24);
            v50 = v36(v31, v30);
            v37 = v60;
            v38 = v50;
          }
          *(_DWORD *)(v3[15] + 4LL * v37) += v38;
        }
      }
    }
  }
}
