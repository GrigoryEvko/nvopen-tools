// Function: sub_3599870
// Address: 0x3599870
//
void __fastcall sub_3599870(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        int a5,
        __int64 a6,
        int a7,
        int a8,
        int a9)
{
  __int64 v12; // rdi
  int v13; // eax
  __int64 v14; // rdx
  __int64 v15; // r12
  __int64 i; // rbx
  __int64 v17; // r13
  __int64 v18; // rcx
  __int64 v19; // rsi
  unsigned int v20; // edx
  __int64 *v21; // rax
  __int64 v22; // rdi
  int v23; // eax
  __int64 v24; // rcx
  int v25; // r15d
  unsigned int v26; // edi
  __int64 v27; // rsi
  unsigned int v28; // eax
  __int64 *v29; // rdx
  __int64 v30; // r8
  __int64 *v31; // rax
  int v32; // r8d
  int v33; // r9d
  __int64 v34; // r8
  int v35; // edx
  unsigned int v36; // r11d
  unsigned int v37; // r9d
  __int64 *v38; // rdi
  __int64 v39; // r10
  int v40; // r9d
  int v41; // eax
  int v42; // edx
  char v43; // al
  __int32 v44; // eax
  unsigned __int8 *v45; // rsi
  unsigned int v46; // r9d
  __int32 v47; // r8d
  __int64 v48; // rax
  __int64 v49; // rcx
  _QWORD *v50; // rax
  __int64 v51; // rdx
  int v52; // r9d
  int v53; // r10d
  int v54; // edi
  int v55; // [rsp+4h] [rbp-BCh]
  __int32 v56; // [rsp+4h] [rbp-BCh]
  __int64 v60; // [rsp+30h] [rbp-90h]
  int v61; // [rsp+30h] [rbp-90h]
  __int64 v62; // [rsp+30h] [rbp-90h]
  __int32 v63; // [rsp+30h] [rbp-90h]
  __int64 v64; // [rsp+38h] [rbp-88h]
  unsigned int v65; // [rsp+38h] [rbp-88h]
  __int64 v66; // [rsp+38h] [rbp-88h]
  int v67; // [rsp+38h] [rbp-88h]
  int v68; // [rsp+38h] [rbp-88h]
  unsigned int v69; // [rsp+40h] [rbp-80h]
  int v70; // [rsp+44h] [rbp-7Ch]
  unsigned __int8 *v72; // [rsp+58h] [rbp-68h] BYREF
  __int64 v73[2]; // [rsp+60h] [rbp-60h] BYREF
  __int64 v74[10]; // [rsp+70h] [rbp-50h] BYREF

  v12 = *a1;
  v69 = *(_DWORD *)(v12 + 96) - 1;
  v13 = sub_3598DB0(v12, a6);
  v14 = a1[3];
  v70 = a5 + v13;
  if ( a7 < 0 )
    v15 = *(_QWORD *)(*(_QWORD *)(v14 + 56) + 16LL * (a7 & 0x7FFFFFFF) + 8);
  else
    v15 = *(_QWORD *)(*(_QWORD *)(v14 + 304) + 8LL * (unsigned int)a7);
  if ( v15 )
  {
    if ( (*(_BYTE *)(v15 + 3) & 0x10) == 0 )
      goto LABEL_5;
    do
    {
      v15 = *(_QWORD *)(v15 + 32);
      if ( !v15 )
        return;
    }
    while ( (*(_BYTE *)(v15 + 3) & 0x10) != 0 );
    while ( 1 )
    {
LABEL_5:
      for ( i = *(_QWORD *)(v15 + 32); i; i = *(_QWORD *)(i + 32) )
      {
        if ( (*(_BYTE *)(i + 3) & 0x10) == 0 )
          break;
      }
      v17 = *(_QWORD *)(v15 + 16);
      if ( a2 != *(_QWORD *)(v17 + 24)
        || (!*(_WORD *)(v17 + 68) || *(_WORD *)(v17 + 68) == 68)
        && (*(_WORD *)(a6 + 68) && *(_WORD *)(a6 + 68) != 68 && *(_DWORD *)(*(_QWORD *)(v17 + 32) + 8LL) == a8
         || (unsigned int)sub_3598190(*(_QWORD *)(v15 + 16), a2) != a7) )
      {
        goto LABEL_10;
      }
      v18 = *(unsigned int *)(a3 + 24);
      v19 = *(_QWORD *)(a3 + 8);
      if ( (_DWORD)v18 )
      {
        v20 = (v18 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v21 = (__int64 *)(v19 + 16LL * v20);
        v22 = *v21;
        if ( v17 == *v21 )
          goto LABEL_19;
        v41 = 1;
        while ( v22 != -4096 )
        {
          v52 = v41 + 1;
          v20 = (v18 - 1) & (v41 + v20);
          v21 = (__int64 *)(v19 + 16LL * v20);
          v22 = *v21;
          if ( v17 == *v21 )
            goto LABEL_19;
          v41 = v52;
        }
      }
      v21 = (__int64 *)(v19 + 16 * v18);
LABEL_19:
      v60 = v21[1];
      v64 = *a1;
      v23 = sub_3598DB0(*a1, v60);
      v24 = v60;
      v25 = v23;
      v26 = *(_DWORD *)(v64 + 56);
      v27 = *(_QWORD *)(v64 + 40);
      if ( !v26 )
        goto LABEL_60;
      v28 = (v26 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
      v29 = (__int64 *)(v27 + 16LL * v28);
      v30 = *v29;
      if ( v60 != *v29 )
      {
        v42 = 1;
        while ( v30 != -4096 )
        {
          v53 = v42 + 1;
          v28 = (v26 - 1) & (v42 + v28);
          v29 = (__int64 *)(v27 + 16LL * v28);
          v30 = *v29;
          if ( v60 == *v29 )
            goto LABEL_21;
          v42 = v53;
        }
LABEL_60:
        v32 = -1;
        v31 = (__int64 *)(v27 + 16LL * v26);
        goto LABEL_23;
      }
LABEL_21:
      v31 = (__int64 *)(v27 + 16LL * v26);
      if ( v31 == v29 )
        v32 = -1;
      else
        v32 = *((_DWORD *)v29 + 2);
LABEL_23:
      if ( v70 != v25 )
      {
        if ( v69 > a4 )
        {
          v33 = 0;
          if ( v70 <= v25 )
            goto LABEL_10;
LABEL_26:
          if ( *(_WORD *)(a6 + 68) == 68 || !*(_WORD *)(a6 + 68) )
LABEL_28:
            v33 = a8;
LABEL_29:
          if ( !v33 )
            goto LABEL_10;
LABEL_30:
          v65 = v33;
          if ( sub_2EBE590(
                 a1[3],
                 v33,
                 *(_QWORD *)(*(_QWORD *)(a1[3] + 56) + 16LL * (a7 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
                 0) )
          {
            sub_2EAB0C0(v15, v65);
          }
          else
          {
            v44 = sub_2EC06C0(
                    a1[3],
                    *(_QWORD *)(*(_QWORD *)(a1[3] + 56) + 16LL * (a7 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
                    byte_3F871B3,
                    0,
                    v34,
                    v65);
            v45 = *(unsigned __int8 **)(v17 + 56);
            v46 = v65;
            v47 = v44;
            v48 = a1[4];
            v72 = v45;
            v49 = *(_QWORD *)(v48 + 8) - 800LL;
            if ( v45 )
            {
              v56 = v47;
              v62 = *(_QWORD *)(v48 + 8) - 800LL;
              sub_B96E90((__int64)&v72, (__int64)v45, 1);
              v46 = v65;
              v49 = v62;
              v47 = v56;
              v74[0] = (__int64)v72;
              if ( v72 )
              {
                sub_B976B0((__int64)&v72, v72, (__int64)v74);
                v72 = 0;
                v47 = v56;
                v49 = v62;
                v46 = v65;
              }
            }
            else
            {
              v74[0] = 0;
            }
            v67 = v47;
            v63 = v46;
            v74[1] = 0;
            v74[2] = 0;
            v50 = sub_2F2A600(a2, v17, v74, v49, v47);
            v73[1] = v51;
            v73[0] = (__int64)v50;
            sub_3598AB0(v73, v63, 0, 0);
            sub_9C6650(v74);
            sub_9C6650(&v72);
            sub_2EAB0C0(v15, v67);
          }
          goto LABEL_10;
        }
        goto LABEL_40;
      }
      v35 = *(unsigned __int16 *)(a6 + 68);
      if ( *(_WORD *)(a6 + 68) && v35 != 68 )
      {
        if ( v69 > a4 )
          goto LABEL_10;
LABEL_40:
        v33 = 0;
        if ( v70 + 1 == v25 )
        {
          v43 = sub_3599670(a1, a6);
          v33 = 0;
          if ( !v43 )
            v33 = a8;
        }
        if ( v70 > v25 )
          goto LABEL_26;
        goto LABEL_42;
      }
      if ( v26 )
      {
        v36 = v26 - 1;
        v37 = (v26 - 1) & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4));
        v38 = (__int64 *)(v27 + 16LL * v37);
        v39 = *v38;
        if ( a6 == *v38 )
        {
LABEL_50:
          if ( v31 != v38 )
          {
            v40 = *((_DWORD *)v38 + 2);
            goto LABEL_52;
          }
        }
        else
        {
          v54 = 1;
          while ( v39 != -4096 )
          {
            v37 = v36 & (v54 + v37);
            v68 = v54 + 1;
            v38 = (__int64 *)(v27 + 16LL * v37);
            v39 = *v38;
            if ( a6 == *v38 )
              goto LABEL_50;
            v54 = v68;
          }
        }
      }
      v40 = -1;
LABEL_52:
      if ( a9 )
      {
        v55 = v40;
        v61 = v32;
        v66 = v24;
        if ( v69 > a4 )
        {
          v33 = a9;
          goto LABEL_30;
        }
        if ( !(unsigned __int8)sub_3599670(a1, a6)
          && (v61 >= v55 || !*(_WORD *)(v66 + 68) || *(_WORD *)(v66 + 68) == 68) )
        {
          v33 = a9;
LABEL_42:
          v35 = *(unsigned __int16 *)(a6 + 68);
          goto LABEL_43;
        }
        v35 = *(unsigned __int16 *)(a6 + 68);
      }
      else if ( v69 > a4 )
      {
        goto LABEL_28;
      }
      v33 = a8;
LABEL_43:
      if ( v35 == 0 || v35 == 68 )
        goto LABEL_29;
      if ( v70 < v25 )
        v33 = a8;
      if ( v33 )
        goto LABEL_30;
LABEL_10:
      if ( !i )
        return;
      v15 = i;
    }
  }
}
