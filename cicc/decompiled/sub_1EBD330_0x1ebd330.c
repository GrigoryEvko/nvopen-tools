// Function: sub_1EBD330
// Address: 0x1ebd330
//
__int64 __fastcall sub_1EBD330(__int64 a1, __int64 a2, unsigned int a3, unsigned __int8 a4, float *a5)
{
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdx
  unsigned int v10; // r13d
  unsigned int v11; // ecx
  __int16 v12; // ax
  _WORD *v13; // rcx
  __int16 *v14; // rdx
  int v15; // eax
  __int64 v16; // r14
  __int64 v17; // r12
  __int64 v18; // rax
  _DWORD *v19; // rdx
  char v20; // dl
  unsigned __int8 v21; // al
  float v22; // esi
  __int16 v24; // ax
  __int64 v25; // r9
  __int64 v26; // rdi
  __int64 v27; // r11
  unsigned __int64 v28; // r10
  int *v29; // r8
  int v30; // edx
  unsigned int v31; // r10d
  _DWORD *v32; // r9
  int *v33; // [rsp+0h] [rbp-70h]
  _DWORD *v34; // [rsp+0h] [rbp-70h]
  __int16 *v35; // [rsp+8h] [rbp-68h]
  __int64 v36; // [rsp+10h] [rbp-60h]
  unsigned __int16 v38; // [rsp+1Ch] [rbp-54h]
  __int64 v40; // [rsp+20h] [rbp-50h]
  char v42; // [rsp+30h] [rbp-40h]
  unsigned int v43; // [rsp+30h] [rbp-40h]
  unsigned int v44; // [rsp+38h] [rbp-38h]
  float v45; // [rsp+3Ch] [rbp-34h]

  v36 = sub_1DBCA20(*(_QWORD *)(a1 + 264), a2);
  v44 = *(_DWORD *)(*(_QWORD *)(a1 + 920) + 8LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF) + 4);
  if ( !v44 )
    v44 = *(_DWORD *)(a1 + 912);
  v9 = *(_QWORD *)(a1 + 696);
  if ( !v9 )
    BUG();
  v10 = 0;
  v45 = 0.0;
  v11 = *(_DWORD *)(*(_QWORD *)(v9 + 8) + 24LL * a3 + 16);
  v12 = a3 * (v11 & 0xF);
  v13 = (_WORD *)(*(_QWORD *)(v9 + 56) + 2LL * (v11 >> 4));
  v14 = v13 + 1;
  v38 = *v13 + v12;
LABEL_5:
  v35 = v14;
  while ( v35 )
  {
    v40 = sub_2103840(*(_QWORD *)(a1 + 272), a2, v38, v13, v7, v8);
    if ( (unsigned int)sub_20FD0B0(v40, 10) > 9 )
      return 0;
    v15 = *(_DWORD *)(v40 + 120);
    if ( v15 )
    {
      v16 = 8LL * (unsigned int)(v15 - 1);
      do
      {
        v17 = *(_QWORD *)(*(_QWORD *)(v40 + 112) + v16);
        v18 = *(_DWORD *)(v17 + 112) & 0x7FFFFFFF;
        v19 = (_DWORD *)(*(_QWORD *)(a1 + 920) + 8 * v18);
        if ( *v19 == 6 )
          return 0;
        if ( INFINITY != *(float *)(a2 + 116) )
          goto LABEL_43;
        if ( INFINITY != *(float *)(v17 + 116) )
          goto LABEL_44;
        v25 = *(_QWORD *)(a1 + 280);
        v26 = a1 + 280;
        v27 = *(_QWORD *)(*(_QWORD *)(a1 + 248) + 24LL);
        v28 = *(_QWORD *)(v27 + 16LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
        v29 = (int *)(v25 + 24LL * *(unsigned __int16 *)(*(_QWORD *)v28 + 24LL));
        v30 = *v29;
        if ( *(_DWORD *)(a1 + 288) != *v29 )
        {
          v33 = (int *)(v25 + 24LL * *(unsigned __int16 *)(*(_QWORD *)v28 + 24LL));
          sub_1ED7890(v26);
          v25 = *(_QWORD *)(a1 + 280);
          v29 = v33;
          v27 = *(_QWORD *)(*(_QWORD *)(a1 + 248) + 24LL);
          v30 = *(_DWORD *)(a1 + 288);
          v26 = a1 + 280;
          v18 = *(_DWORD *)(v17 + 112) & 0x7FFFFFFF;
        }
        v31 = v29[1];
        v32 = (_DWORD *)(v25
                       + 24LL
                       * *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(v27 + 16 * v18) & 0xFFFFFFFFFFFFFFF8LL) + 24LL));
        if ( *v32 != v30 )
        {
          v34 = v32;
          v43 = v29[1];
          sub_1ED7890(v26);
          v32 = v34;
          v31 = v43;
          v18 = *(_DWORD *)(v17 + 112) & 0x7FFFFFFF;
        }
        v19 = (_DWORD *)(*(_QWORD *)(a1 + 920) + 8 * v18);
        if ( v31 < v32[1] )
        {
LABEL_44:
          if ( v44 <= v19[1] )
            v10 += 10;
          v20 = 1;
        }
        else
        {
LABEL_43:
          if ( v44 <= v19[1] )
            return 0;
          v20 = 0;
        }
        v42 = v20;
        v21 = sub_1F5BE30(*(_QWORD *)(a1 + 256));
        v10 += v21;
        v22 = *a5;
        v45 = fmaxf(*(float *)(v17 + 116), v45);
        if ( v10 >= *(_DWORD *)a5 && (v10 != LODWORD(v22) || a5[1] <= v45) )
          return 0;
        if ( !v42
          && ((((*(_DWORD *)(*(_QWORD *)(a1 + 920) + 8LL * (*(_DWORD *)(v17 + 112) & 0x7FFFFFFF)) <= 3) & a4) == 0 || v21)
           && *(float *)(a2 + 116) <= *(float *)(v17 + 116)
           || v36
           && v22 != NAN
           && sub_1DBCA20(*(_QWORD *)(a1 + 264), v17)
           && (!*(_BYTE *)(a1 + 27408) || !(unsigned int)sub_1EBBEC0((_QWORD *)a1, v17, a3))) )
        {
          return 0;
        }
        v16 -= 8;
      }
      while ( v16 != -8 );
    }
    v14 = 0;
    v24 = *v35;
    v13 = v35 + 1;
    v38 += *v35++;
    if ( !v24 )
      goto LABEL_5;
  }
  *(_DWORD *)a5 = v10;
  a5[1] = v45;
  return 1;
}
