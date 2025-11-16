// Function: sub_DA4700
// Address: 0xda4700
//
__int64 __fastcall sub_DA4700(
        _QWORD *a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        unsigned int a6)
{
  int v6; // ebx
  int v7; // eax
  unsigned __int64 v8; // r12
  _BYTE *v10; // rdx
  _QWORD *v12; // rsi
  _QWORD *v14; // rdx
  _QWORD *v15; // rax
  _QWORD *v16; // r15
  __int64 v17; // rdi
  __int64 v18; // rcx
  _BYTE *v19; // rcx
  _QWORD *v20; // rax
  _QWORD *v21; // r15
  __int64 v22; // rsi
  _BYTE *v23; // r10
  _QWORD *v24; // rax
  _QWORD *v25; // r11
  _BYTE *v26; // rdi
  _BYTE *v27; // rax
  _BYTE *v28; // r10
  _BYTE *v29; // rsi
  _QWORD *v30; // rax
  _QWORD *v31; // r11
  _BYTE *v32; // rdi
  _BYTE *v33; // rax
  __int64 v34; // rax
  unsigned int v35; // edx
  unsigned int v36; // ebx
  int v37; // edx
  __int64 v38; // r15
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rax
  int v42; // ebx
  __int64 v43; // rdx
  __int64 v44; // rcx
  int v45; // eax
  int v46; // esi
  unsigned int v47; // r9d
  __int64 v48; // rax
  __int64 v49; // r12
  __int64 v50; // rbx
  __int64 v51; // r15
  _QWORD *v52; // [rsp+8h] [rbp-88h]
  _QWORD *v53; // [rsp+8h] [rbp-88h]
  unsigned int v54; // [rsp+14h] [rbp-7Ch]
  unsigned int v55; // [rsp+14h] [rbp-7Ch]
  __int64 v56; // [rsp+18h] [rbp-78h]
  __int64 v57; // [rsp+18h] [rbp-78h]
  _BYTE *v58; // [rsp+20h] [rbp-70h]
  _BYTE *v59; // [rsp+28h] [rbp-68h]
  unsigned __int64 v60; // [rsp+28h] [rbp-68h]
  _BYTE *v61; // [rsp+30h] [rbp-60h]
  _QWORD *v62; // [rsp+30h] [rbp-60h]
  __int64 v63; // [rsp+30h] [rbp-60h]
  _BYTE *v64; // [rsp+38h] [rbp-58h]
  _BYTE *v65; // [rsp+38h] [rbp-58h]
  unsigned int v66; // [rsp+38h] [rbp-58h]
  __int64 v67; // [rsp+38h] [rbp-58h]
  _QWORD *v68; // [rsp+40h] [rbp-50h]
  _QWORD *v69; // [rsp+40h] [rbp-50h]
  __int64 v70; // [rsp+40h] [rbp-50h]
  unsigned int v71; // [rsp+40h] [rbp-50h]
  unsigned __int64 v72; // [rsp+48h] [rbp-48h] BYREF
  __int64 v73; // [rsp+50h] [rbp-40h]
  __int64 v74; // [rsp+58h] [rbp-38h]

  v72 = a4;
  if ( a4 == a3 )
  {
LABEL_9:
    LODWORD(v74) = 0;
    BYTE4(v74) = 1;
    return v74;
  }
  v6 = *(unsigned __int16 *)(a3 + 24);
  v7 = *(unsigned __int16 *)(a4 + 24);
  v8 = a3;
  if ( (_WORD)v6 == (_WORD)v7 )
  {
    v12 = (_QWORD *)a1[2];
    v14 = a1 + 1;
    v15 = v12;
    if ( !v12 )
      goto LABEL_25;
    v16 = a1 + 1;
    do
    {
      while ( 1 )
      {
        v17 = v15[2];
        v18 = v15[3];
        if ( v15[6] >= v8 )
          break;
        v15 = (_QWORD *)v15[3];
        if ( !v18 )
          goto LABEL_15;
      }
      v16 = v15;
      v15 = (_QWORD *)v15[2];
    }
    while ( v17 );
LABEL_15:
    if ( v14 == v16 || v16[6] > v8 )
      goto LABEL_25;
    v19 = v16 + 4;
    if ( (v16[5] & 1) == 0 )
    {
      v19 = (_BYTE *)v16[4];
      if ( (v19[8] & 1) == 0 )
      {
        v28 = *(_BYTE **)v19;
        if ( (*(_BYTE *)(*(_QWORD *)v19 + 8LL) & 1) != 0 )
        {
          v16[4] = v28;
          v19 = v28;
          v12 = (_QWORD *)a1[2];
        }
        else
        {
          v29 = *(_BYTE **)v28;
          if ( (*(_BYTE *)(*(_QWORD *)v28 + 8LL) & 1) == 0 )
          {
            v30 = *(_QWORD **)v29;
            v69 = *(_QWORD **)v29;
            if ( (*(_BYTE *)(*(_QWORD *)v29 + 8LL) & 1) != 0 )
            {
              v29 = *(_BYTE **)v29;
            }
            else
            {
              v31 = (_QWORD *)*v30;
              if ( (*(_BYTE *)(*v30 + 8LL) & 1) == 0 )
              {
                v32 = (_BYTE *)*v31;
                v53 = (_QWORD *)*v30;
                if ( (*(_BYTE *)(*v31 + 8LL) & 1) == 0 )
                {
                  v55 = a6;
                  v57 = a5;
                  v59 = *(_BYTE **)v19;
                  v62 = v14;
                  v65 = (_BYTE *)v16[4];
                  v33 = sub_D9F6F0(v32);
                  a6 = v55;
                  a5 = v57;
                  v32 = v33;
                  v28 = v59;
                  v14 = v62;
                  *v53 = v33;
                  v19 = v65;
                }
                v31 = v32;
                *v69 = v32;
              }
              *(_QWORD *)v29 = v31;
              v29 = v31;
            }
            *(_QWORD *)v28 = v29;
          }
          *(_QWORD *)v19 = v29;
          v16[4] = v29;
          if ( !v29 )
            goto LABEL_25;
          v19 = v29;
          v12 = (_QWORD *)a1[2];
        }
      }
    }
    v20 = v12;
    if ( v12 )
    {
      v21 = v14;
      do
      {
        if ( v20[6] < v72 )
        {
          v20 = (_QWORD *)v20[3];
        }
        else
        {
          v21 = v20;
          v20 = (_QWORD *)v20[2];
        }
      }
      while ( v20 );
      if ( v14 != v21 && v21[6] <= v72 )
      {
        v10 = v21 + 4;
        if ( (v21[5] & 1) == 0 )
        {
          v10 = (_BYTE *)v21[4];
          if ( (v10[8] & 1) == 0 )
          {
            v22 = *(_QWORD *)v10;
            if ( (*(_BYTE *)(*(_QWORD *)v10 + 8LL) & 1) != 0 )
            {
              v10 = *(_BYTE **)v10;
            }
            else
            {
              v23 = *(_BYTE **)v22;
              if ( (*(_BYTE *)(*(_QWORD *)v22 + 8LL) & 1) == 0 )
              {
                v24 = *(_QWORD **)v23;
                v68 = *(_QWORD **)v23;
                if ( (*(_BYTE *)(*(_QWORD *)v23 + 8LL) & 1) != 0 )
                {
                  v23 = *(_BYTE **)v23;
                }
                else
                {
                  v25 = (_QWORD *)*v24;
                  if ( (*(_BYTE *)(*v24 + 8LL) & 1) == 0 )
                  {
                    v26 = (_BYTE *)*v25;
                    v52 = (_QWORD *)*v24;
                    if ( (*(_BYTE *)(*v25 + 8LL) & 1) == 0 )
                    {
                      v54 = a6;
                      v56 = a5;
                      v58 = *(_BYTE **)v22;
                      v61 = (_BYTE *)v21[4];
                      v64 = v19;
                      v27 = sub_D9F6F0(v26);
                      a6 = v54;
                      a5 = v56;
                      v23 = v58;
                      v26 = v27;
                      v10 = v61;
                      *v52 = v27;
                      v19 = v64;
                    }
                    v25 = v26;
                    *v68 = v26;
                  }
                  *(_QWORD *)v23 = v25;
                  v23 = v25;
                }
                *(_QWORD *)v22 = v23;
              }
              *(_QWORD *)v10 = v23;
              v10 = v23;
            }
            v21[4] = v10;
          }
        }
        if ( v19 == v10 )
          goto LABEL_9;
      }
    }
LABEL_25:
    if ( a6 > dword_4F895A8 )
    {
      BYTE4(v74) = 0;
      return v74;
    }
    switch ( (__int16)v6 )
    {
      case 0:
        v43 = *(_QWORD *)(v8 + 32);
        v44 = *(_QWORD *)(v72 + 32);
        v45 = *(_DWORD *)(v43 + 32);
        v46 = *(_DWORD *)(v44 + 32);
        BYTE4(v74) = 1;
        if ( v45 == v46 )
          LODWORD(v74) = ((int)sub_C49970(v43 + 24, (unsigned __int64 *)(v44 + 24)) >> 31) | 1;
        else
          LODWORD(v74) = v45 - v46;
        return v74;
      case 1:
        v41 = *(_QWORD *)(v8 + 32);
        BYTE4(v74) = 1;
        LODWORD(v74) = (*(_DWORD *)(v41 + 8) >> 8) - (*(_DWORD *)(*(_QWORD *)(v72 + 32) + 8LL) >> 8);
        return v74;
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 9:
      case 10:
      case 11:
      case 12:
      case 13:
      case 14:
        goto LABEL_51;
      case 8:
        v39 = *(_QWORD *)(v8 + 48);
        v40 = *(_QWORD *)(v72 + 48);
        if ( v40 != v39 )
        {
          if ( (unsigned __int8)sub_B19720(a5, **(_QWORD **)(v39 + 32), **(_QWORD **)(v40 + 32)) )
            LODWORD(v74) = 1;
          else
            LODWORD(v74) = -1;
          BYTE4(v74) = 1;
          return v74;
        }
LABEL_51:
        v66 = a6;
        v70 = a5;
        v34 = sub_D960E0(v8);
        v36 = v35;
        v63 = v34;
        v38 = sub_D960E0(v72);
        if ( v36 != v37 )
        {
          BYTE4(v74) = 1;
          LODWORD(v74) = v36 - v37;
          return v74;
        }
        if ( v36 )
        {
          v60 = v8;
          v47 = v66 + 1;
          v48 = 8LL * v36;
          v49 = v38;
          v50 = 0;
          v51 = v70;
          v67 = v48;
          do
          {
            v71 = v47;
            v73 = sub_DA4700(a1, a2, *(_QWORD *)(v50 + v63), *(_QWORD *)(v50 + v49), v51);
            if ( BYTE4(v73) != 1 )
              return v73;
            v47 = v71;
            if ( (_DWORD)v73 )
              return v73;
            v50 += 8;
          }
          while ( v67 != v50 );
          v8 = v60;
        }
        sub_DA4500(a1, v8, (__int64 *)&v72);
        break;
      case 15:
        if ( !v72 )
          BUG();
        v42 = sub_D93590(a2, *(unsigned __int8 **)(v8 - 8), *(unsigned __int8 **)(v72 - 8), a6 + 1);
        if ( !v42 )
          sub_DA4500(a1, v8, (__int64 *)&v72);
        LODWORD(v74) = v42;
        BYTE4(v74) = 1;
        return v74;
      default:
        BUG();
    }
    goto LABEL_9;
  }
  BYTE4(v74) = 1;
  LODWORD(v74) = v6 - v7;
  return v74;
}
