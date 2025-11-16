// Function: sub_1E46110
// Address: 0x1e46110
//
void __fastcall sub_1E46110(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        int a6,
        __int64 a7,
        int a8,
        unsigned int a9,
        unsigned int a10)
{
  unsigned __int64 v12; // rax
  int v13; // eax
  __int64 v14; // rdx
  __int64 v16; // r12
  __int64 v17; // r14
  __int64 v18; // rbx
  __int64 i; // rax
  __int64 v20; // rsi
  int v21; // edi
  __int64 v22; // rdx
  int v23; // edx
  __int64 v24; // rcx
  __int64 v25; // rdi
  unsigned int v26; // esi
  __int64 *v27; // rdx
  __int64 v28; // r9
  unsigned __int64 v29; // r12
  int v30; // eax
  _QWORD *v31; // r15
  int v32; // ecx
  _QWORD *v33; // rdx
  _QWORD *v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rsi
  unsigned int v37; // r15d
  _WORD *v38; // rsi
  int v39; // edx
  __int64 v40; // rax
  __int64 v41; // r9
  unsigned __int64 v42; // rdi
  _QWORD *v43; // r11
  __int64 v44; // rsi
  __int64 v45; // rax
  int v46; // r15d
  char v47; // al
  bool v48; // al
  char v49; // al
  int v50; // r8d
  int v51; // edx
  int v52; // eax
  __int64 v53; // [rsp+8h] [rbp-88h]
  __int64 v54; // [rsp+10h] [rbp-80h]
  int v55; // [rsp+10h] [rbp-80h]
  int v56; // [rsp+18h] [rbp-78h]
  int v57; // [rsp+20h] [rbp-70h]
  int v58; // [rsp+20h] [rbp-70h]
  __int64 v59; // [rsp+20h] [rbp-70h]
  bool v60; // [rsp+2Fh] [rbp-61h]
  int v61; // [rsp+38h] [rbp-58h]
  unsigned int v62; // [rsp+3Ch] [rbp-54h]
  __int64 v65; // [rsp+50h] [rbp-40h]
  __int64 v66; // [rsp+50h] [rbp-40h]
  _QWORD *v67; // [rsp+50h] [rbp-40h]
  int v68; // [rsp+50h] [rbp-40h]
  int v69; // [rsp+50h] [rbp-40h]
  int v71; // [rsp+5Ch] [rbp-34h]

  v62 = (*(_DWORD *)(a3 + 132) - *(_DWORD *)(a3 + 128)) / *(_DWORD *)(a3 + 136);
  v60 = a5 < v62;
  v12 = sub_1E45EB0(a1, a7);
  v13 = sub_1E404B0(a3, v12);
  v14 = *(_QWORD *)(a1 + 40);
  v71 = a6 + v13;
  if ( a8 < 0 )
    v16 = *(_QWORD *)(*(_QWORD *)(v14 + 24) + 16LL * (a8 & 0x7FFFFFFF) + 8);
  else
    v16 = *(_QWORD *)(*(_QWORD *)(v14 + 272) + 8LL * (unsigned int)a8);
  if ( v16 )
  {
    if ( (*(_BYTE *)(v16 + 3) & 0x10) == 0 )
    {
LABEL_5:
      v17 = v16;
      v61 = v71 + 1;
      while ( 1 )
      {
        v18 = *(_QWORD *)(v17 + 32);
        for ( i = *(_QWORD *)(v17 + 16); v18; v18 = *(_QWORD *)(v18 + 32) )
        {
          if ( (*(_BYTE *)(v18 + 3) & 0x10) == 0 )
            break;
        }
        if ( a2 != *(_QWORD *)(i + 24) )
          goto LABEL_11;
        if ( !**(_WORD **)(i + 16) || **(_WORD **)(i + 16) == 45 )
        {
          v20 = *(_QWORD *)(i + 32);
          if ( **(_WORD **)(a7 + 16) )
          {
            if ( **(_WORD **)(a7 + 16) != 45 && *(_DWORD *)(v20 + 8) == a9 )
              goto LABEL_11;
          }
          v21 = *(_DWORD *)(i + 40);
          if ( v21 == 1 )
          {
LABEL_49:
            v23 = 0;
          }
          else
          {
            v22 = 1;
            while ( a2 != *(_QWORD *)(v20 + 40LL * (unsigned int)(v22 + 1) + 24) )
            {
              v22 = (unsigned int)(v22 + 2);
              if ( v21 == (_DWORD)v22 )
                goto LABEL_49;
            }
            v23 = *(_DWORD *)(v20 + 40 * v22 + 8);
          }
          if ( a8 != v23 )
            goto LABEL_11;
        }
        v24 = *(unsigned int *)(a4 + 24);
        v25 = *(_QWORD *)(a4 + 8);
        if ( (_DWORD)v24 )
        {
          v26 = (v24 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
          v27 = (__int64 *)(v25 + 16LL * v26);
          v28 = *v27;
          if ( i == *v27 )
            goto LABEL_29;
          v39 = 1;
          while ( v28 != -8 )
          {
            v50 = v39 + 1;
            v26 = (v24 - 1) & (v26 + v39);
            v27 = (__int64 *)(v25 + 16LL * v26);
            v28 = *v27;
            if ( i == *v27 )
              goto LABEL_29;
            v39 = v50;
          }
        }
        v27 = (__int64 *)(v25 + 16 * v24);
LABEL_29:
        v65 = a2;
        v29 = sub_1E45EB0(a1, v27[1]);
        v30 = sub_1E404B0(a3, v29);
        v31 = *(_QWORD **)(a3 + 48);
        a2 = v65;
        v32 = v30;
        v33 = (_QWORD *)(a3 + 40);
        if ( v31 )
        {
          v34 = *(_QWORD **)(a3 + 48);
          do
          {
            while ( 1 )
            {
              v35 = v34[2];
              v36 = v34[3];
              if ( v34[4] >= v29 )
                break;
              v34 = (_QWORD *)v34[3];
              if ( !v36 )
                goto LABEL_34;
            }
            v33 = v34;
            v34 = (_QWORD *)v34[2];
          }
          while ( v35 );
LABEL_34:
          if ( v33 != (_QWORD *)(a3 + 40) && v33[4] > v29 )
            v33 = (_QWORD *)(a3 + 40);
        }
        if ( v71 == v32 )
        {
          v38 = *(_WORD **)(a7 + 16);
          if ( !*v38 || *v38 == 45 )
          {
            v54 = v65;
            v57 = v32;
            v67 = v33;
            v40 = sub_1E45EB0(a1, a7);
            v32 = v57;
            v41 = a3 + 40;
            a2 = v54;
            v42 = v40;
            if ( v31 )
            {
              v43 = (_QWORD *)(a3 + 40);
              do
              {
                while ( 1 )
                {
                  v44 = v31[2];
                  v45 = v31[3];
                  if ( v31[4] >= v42 )
                    break;
                  v31 = (_QWORD *)v31[3];
                  if ( !v45 )
                    goto LABEL_57;
                }
                v43 = v31;
                v31 = (_QWORD *)v31[2];
              }
              while ( v44 );
LABEL_57:
              if ( (_QWORD *)(a3 + 40) != v43 && v43[4] <= v42 )
                v41 = (__int64)v43;
            }
            if ( !a10 )
              goto LABEL_63;
            v37 = a10;
            if ( !v60 )
            {
              v55 = v57;
              v53 = a2;
              v68 = *((_DWORD *)v67 + 10);
              v56 = *(_DWORD *)(a3 + 136);
              v46 = *(_DWORD *)(a3 + 128);
              v58 = *(_DWORD *)(v41 + 40);
              v47 = sub_1E45F30(a3, a1, a7);
              v32 = v55;
              a2 = v53;
              if ( v47 )
                goto LABEL_63;
              v51 = (v68 - v46) % v56;
              v52 = v58 - v46;
              v37 = a10;
              if ( v51 < v52 % v56 )
              {
                if ( **(_WORD **)(*(_QWORD *)(v29 + 8) + 16LL) == 45 || !**(_WORD **)(*(_QWORD *)(v29 + 8) + 16LL) )
                {
                  v37 = a10;
                  goto LABEL_64;
                }
LABEL_63:
                v37 = a9;
              }
LABEL_64:
              if ( a5 >= v62 )
              {
                v38 = *(_WORD **)(a7 + 16);
                goto LABEL_41;
              }
LABEL_44:
              if ( !v37 )
                goto LABEL_11;
            }
            v66 = a2;
            sub_1E69410(
              *(_QWORD *)(a1 + 40),
              v37,
              *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 24LL) + 16LL * (a8 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
              0);
            sub_1E310D0(v17, v37);
            a2 = v66;
            goto LABEL_11;
          }
          if ( a5 >= v62 )
          {
            v37 = 0;
            if ( v61 == v32 )
              goto LABEL_74;
            goto LABEL_40;
          }
        }
        else
        {
          if ( a5 >= v62 )
          {
            v37 = 0;
            v38 = *(_WORD **)(a7 + 16);
            if ( v61 == v32 )
            {
LABEL_74:
              if ( *v38 != 45 && *v38 )
              {
                v37 = a9;
                if ( v71 <= v61 )
                  goto LABEL_41;
                goto LABEL_44;
              }
              v59 = v65;
              v69 = v32;
              v49 = sub_1E45F30(a3, a1, a7);
              v32 = v69;
              a2 = v59;
              if ( v49 )
                v37 = 0;
              else
                v37 = a9;
              v38 = *(_WORD **)(a7 + 16);
            }
LABEL_40:
            if ( v71 <= v32 )
            {
LABEL_41:
              if ( *v38 != 45 && *v38 != 0 && v71 < v32 )
                v37 = a9;
              goto LABEL_44;
            }
            v48 = *v38 == 0 || *v38 == 45;
LABEL_68:
            if ( v48 )
              v37 = a9;
            goto LABEL_44;
          }
          if ( v71 > v32 )
          {
            v37 = 0;
            v48 = **(_WORD **)(a7 + 16) == 45 || **(_WORD **)(a7 + 16) == 0;
            goto LABEL_68;
          }
        }
LABEL_11:
        if ( !v18 )
          return;
        v17 = v18;
      }
    }
    while ( 1 )
    {
      v16 = *(_QWORD *)(v16 + 32);
      if ( !v16 )
        break;
      if ( (*(_BYTE *)(v16 + 3) & 0x10) == 0 )
        goto LABEL_5;
    }
  }
}
