// Function: sub_13C13B0
// Address: 0x13c13b0
//
__int64 __fastcall sub_13C13B0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, _QWORD *a5)
{
  unsigned __int64 v5; // r15
  __int64 v10; // r9
  unsigned __int8 v11; // al
  __int16 v12; // dx
  _QWORD *v14; // r8
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rcx
  char v17; // si
  unsigned __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rsi
  _QWORD *v29; // rax
  _QWORD *v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rax
  unsigned __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rsi
  _QWORD *v39; // rax
  _QWORD *v40; // rdi
  unsigned int v41; // r8d
  _QWORD *v42; // rcx
  _QWORD *v43; // rax
  _QWORD *v44; // rdi
  unsigned int v45; // r8d
  _QWORD *v46; // rcx
  __int64 v47; // [rsp+0h] [rbp-50h]
  unsigned __int64 v48; // [rsp+0h] [rbp-50h]
  __int64 v49; // [rsp+0h] [rbp-50h]
  unsigned __int64 v50; // [rsp+0h] [rbp-50h]
  unsigned __int64 v51; // [rsp+8h] [rbp-48h]
  int v52; // [rsp+8h] [rbp-48h]
  unsigned __int64 v53; // [rsp+8h] [rbp-48h]
  int v54; // [rsp+8h] [rbp-48h]
  __int64 v55; // [rsp+10h] [rbp-40h]
  __int64 v56; // [rsp+10h] [rbp-40h]
  __int64 v57; // [rsp+10h] [rbp-40h]
  __int64 v58; // [rsp+10h] [rbp-40h]
  unsigned __int64 v59; // [rsp+10h] [rbp-40h]

  if ( *(_BYTE *)(*a2 + 8LL) != 15 )
    return 1;
  v5 = a2[1];
  if ( v5 )
  {
    while ( 1 )
    {
      v10 = sub_1648700(v5);
      v11 = *(_BYTE *)(v10 + 16);
      if ( v11 <= 0x17u )
      {
        if ( v11 == 5 )
        {
          v12 = *(_WORD *)(v10 + 18);
          if ( v12 == 32 )
            goto LABEL_9;
          if ( v12 == 47 )
          {
LABEL_47:
            v14 = a5;
LABEL_10:
            if ( (unsigned __int8)sub_13C13B0(a1, v10, a3, a4, v14) )
              return 1;
            goto LABEL_11;
          }
        }
        if ( (unsigned __int8)(v11 - 4) > 0xCu || (unsigned __int8)sub_1593E70(v10) )
          return 1;
        goto LABEL_11;
      }
      if ( v11 == 54 )
      {
        if ( a3 )
        {
          v28 = *(_QWORD *)(*(_QWORD *)(v10 + 40) + 56LL);
          v29 = *(_QWORD **)(a3 + 8);
          if ( *(_QWORD **)(a3 + 16) != v29 )
            goto LABEL_37;
          v44 = &v29[*(unsigned int *)(a3 + 28)];
          v45 = *(_DWORD *)(a3 + 28);
          if ( v29 != v44 )
          {
            v46 = 0;
            while ( v28 != *v29 )
            {
              if ( *v29 == -2 )
                v46 = v29;
              if ( v44 == ++v29 )
              {
                if ( !v46 )
                  goto LABEL_90;
                *v46 = v28;
                --*(_DWORD *)(a3 + 32);
                ++*(_QWORD *)a3;
                goto LABEL_11;
              }
            }
            goto LABEL_11;
          }
LABEL_90:
          if ( v45 < *(_DWORD *)(a3 + 24) )
          {
            *(_DWORD *)(a3 + 28) = v45 + 1;
            *v44 = v28;
            ++*(_QWORD *)a3;
          }
          else
          {
LABEL_37:
            sub_16CCBA0(a3, v28);
          }
        }
        goto LABEL_11;
      }
      if ( v11 != 55 )
        break;
      v30 = *(_QWORD **)(v10 - 24);
      if ( v30 && a2 == v30 )
      {
        if ( a4 )
        {
          v38 = *(_QWORD *)(*(_QWORD *)(v10 + 40) + 56LL);
          v43 = *(_QWORD **)(a4 + 8);
          if ( *(_QWORD **)(a4 + 16) == v43 )
          {
            v40 = &v43[*(unsigned int *)(a4 + 28)];
            v41 = *(_DWORD *)(a4 + 28);
            if ( v43 != v40 )
            {
              v42 = 0;
              while ( v38 != *v43 )
              {
                if ( *v43 == -2 )
                  v42 = v43;
                if ( v40 == ++v43 )
                {
LABEL_66:
                  if ( !v42 )
                    goto LABEL_92;
                  *v42 = v38;
                  --*(_DWORD *)(a4 + 32);
                  ++*(_QWORD *)a4;
                  goto LABEL_11;
                }
              }
              goto LABEL_11;
            }
LABEL_92:
            if ( v41 < *(_DWORD *)(a4 + 24) )
            {
              *(_DWORD *)(a4 + 28) = v41 + 1;
              *v40 = v38;
              ++*(_QWORD *)a4;
              goto LABEL_11;
            }
          }
LABEL_70:
          sub_16CCBA0(a4, v38);
        }
      }
      else if ( a5 != v30 )
      {
        return 1;
      }
LABEL_11:
      v5 = *(_QWORD *)(v5 + 8);
      if ( !v5 )
        return 0;
    }
    switch ( v11 )
    {
      case '8':
LABEL_9:
        v14 = 0;
        goto LABEL_10;
      case 'G':
        goto LABEL_47;
      case 'N':
        v15 = v10 | 4;
        break;
      default:
        v15 = v10 & 0xFFFFFFFFFFFFFFFBLL;
        if ( v11 != 29 )
        {
          if ( v11 != 75 || *(_BYTE *)(*(_QWORD *)(v10 - 24) + 16LL) != 15 )
            return 1;
          goto LABEL_11;
        }
        break;
    }
    v16 = v15 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v15 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      return 1;
    v17 = *(_BYTE *)(v16 + 23);
    if ( (v17 & 0x40) != 0 )
    {
      v18 = *(_QWORD *)(v16 - 8);
      if ( v18 > v5 )
        goto LABEL_11;
      v19 = 24LL * (*(_DWORD *)(v16 + 20) & 0xFFFFFFF);
      if ( (v15 & 4) == 0 )
      {
LABEL_27:
        if ( v5 >= v18 + v19 - 72 )
          goto LABEL_11;
        if ( v5 < v16 - v19 )
          return 1;
        if ( v17 < 0 )
        {
          v47 = v10;
          v51 = v15 & 0xFFFFFFFFFFFFFFF8LL;
          v20 = sub_1648A40(v16);
          v16 = v51;
          v10 = v47;
          v55 = v21 + v20;
          if ( *(char *)(v51 + 23) >= 0 )
          {
            if ( (unsigned int)(v55 >> 4) )
LABEL_95:
              BUG();
          }
          else
          {
            v22 = sub_1648A40(v51);
            v16 = v51;
            v10 = v47;
            if ( (unsigned int)((v55 - v22) >> 4) )
            {
              v56 = v47;
              if ( *(char *)(v51 + 23) >= 0 )
                goto LABEL_95;
              v48 = v51;
              v23 = sub_1648A40(v51);
              v24 = v51;
              v52 = *(_DWORD *)(v23 + 8);
              if ( *(char *)(v24 + 23) >= 0 )
                BUG();
              v25 = sub_1648A40(v48);
              v10 = v56;
              v16 = v48;
              v27 = -72 - 24LL * (unsigned int)(*(_DWORD *)(v25 + v26 - 4) - v52);
              goto LABEL_56;
            }
          }
        }
        v27 = -72;
        goto LABEL_56;
      }
    }
    else
    {
      v19 = 24LL * (*(_DWORD *)(v16 + 20) & 0xFFFFFFF);
      v18 = v16 - v19;
      if ( v16 - v19 > v5 )
        goto LABEL_11;
      if ( (v15 & 4) == 0 )
        goto LABEL_27;
    }
    if ( v5 >= v18 + v19 - 24 )
      goto LABEL_11;
    if ( v5 < v16 - v19 )
      return 1;
    if ( v17 < 0 )
    {
      v49 = v10;
      v53 = v15 & 0xFFFFFFFFFFFFFFF8LL;
      v31 = sub_1648A40(v16);
      v16 = v53;
      v10 = v49;
      v57 = v32 + v31;
      if ( *(char *)(v53 + 23) >= 0 )
      {
        if ( (unsigned int)(v57 >> 4) )
LABEL_98:
          BUG();
      }
      else
      {
        v33 = sub_1648A40(v53);
        v16 = v53;
        v10 = v49;
        if ( (unsigned int)((v57 - v33) >> 4) )
        {
          v58 = v49;
          if ( *(char *)(v53 + 23) >= 0 )
            goto LABEL_98;
          v50 = v53;
          v34 = sub_1648A40(v53);
          v35 = v53;
          v54 = *(_DWORD *)(v34 + 8);
          if ( *(char *)(v35 + 23) >= 0 )
            BUG();
          v36 = sub_1648A40(v50);
          v10 = v58;
          v16 = v50;
          v27 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v36 + v37 - 4) - v54);
LABEL_56:
          v59 = v16;
          if ( v16 + v27 <= v5 || !sub_140B650(v10, *(_QWORD *)(a1 + 16)) )
            return 1;
          if ( a4 )
          {
            v38 = *(_QWORD *)(*(_QWORD *)(v59 + 40) + 56LL);
            v39 = *(_QWORD **)(a4 + 8);
            if ( *(_QWORD **)(a4 + 16) == v39 )
            {
              v40 = &v39[*(unsigned int *)(a4 + 28)];
              v41 = *(_DWORD *)(a4 + 28);
              if ( v39 != v40 )
              {
                v42 = 0;
                while ( v38 != *v39 )
                {
                  if ( *v39 == -2 )
                    v42 = v39;
                  if ( v40 == ++v39 )
                    goto LABEL_66;
                }
                goto LABEL_11;
              }
              goto LABEL_92;
            }
            goto LABEL_70;
          }
          goto LABEL_11;
        }
      }
    }
    v27 = -24;
    goto LABEL_56;
  }
  return 0;
}
