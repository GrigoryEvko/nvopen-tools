// Function: sub_1F05010
// Address: 0x1f05010
//
__int64 __fastcall sub_1F05010(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _BYTE *a6)
{
  __int64 v6; // r8
  __int16 v8; // ax
  _QWORD *v9; // r13
  _QWORD *v10; // r15
  int v11; // r11d
  __int64 v12; // r14
  unsigned __int64 v13; // rdx
  unsigned int v14; // ebx
  _DWORD *v15; // rsi
  unsigned int v16; // edx
  unsigned int v17; // ecx
  __int64 v18; // rax
  unsigned int v19; // edi
  _DWORD *v20; // rsi
  _DWORD *v21; // r10
  unsigned int v22; // esi
  _DWORD *v23; // rdi
  _DWORD *v24; // rsi
  __int64 v25; // rax
  __int64 v26; // r8
  unsigned int v27; // esi
  int v28; // ecx
  _BYTE *v29; // r8
  unsigned int v30; // eax
  __int64 v31; // rdi
  unsigned int *v32; // rdx
  _BYTE *v34; // rax
  _BYTE *v35; // rdi
  int v36; // edx
  unsigned int v37; // r8d
  unsigned int v38; // esi
  _BYTE *v39; // rdi
  unsigned int v40; // edx
  __int64 v41; // rax
  _DWORD *v42; // rcx
  _BYTE *v43; // r14
  unsigned int v44; // esi
  unsigned int v45; // edx
  _DWORD *v46; // rcx
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rcx
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rax
  int v60; // [rsp+4h] [rbp-5Ch]
  int v61; // [rsp+8h] [rbp-58h]
  int v62; // [rsp+8h] [rbp-58h]
  int v63; // [rsp+8h] [rbp-58h]
  unsigned int v65; // [rsp+18h] [rbp-48h]
  unsigned int v66; // [rsp+1Ch] [rbp-44h]
  unsigned __int64 v67; // [rsp+24h] [rbp-3Ch]

  v6 = a2;
  v65 = *(_DWORD *)(a2 + 192);
  *(_DWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 8LL * v65 + 4) = v65;
  v8 = **(_WORD **)(*(_QWORD *)(a2 + 8) + 16LL);
  switch ( v8 )
  {
    case 0:
    case 8:
    case 10:
    case 14:
    case 15:
    case 45:
LABEL_3:
      v66 = 0;
      break;
    default:
      switch ( v8 )
      {
        case 2:
        case 3:
        case 4:
        case 6:
        case 9:
        case 12:
        case 13:
        case 17:
        case 18:
          goto LABEL_3;
        default:
          v66 = 1;
          break;
      }
      break;
  }
  v9 = *(_QWORD **)(a2 + 32);
  v10 = &v9[2 * *(unsigned int *)(a2 + 40)];
  v11 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 8LL * v65);
  if ( v9 == v10 )
    goto LABEL_26;
  do
  {
    if ( (*v9 & 6) != 0 )
      goto LABEL_24;
    v12 = *(unsigned int *)((*v9 & 0xFFFFFFFFFFFFFFF8LL) + 192);
    v13 = *v9 & 0xFFFFFFFFFFFFFFF8LL;
    v14 = *(_DWORD *)(v13 + 192);
    v15 = (_DWORD *)(8 * v12 + *(_QWORD *)(*(_QWORD *)a1 + 8LL));
    if ( (unsigned int)(v11 - *v15) < *(_DWORD *)(*(_QWORD *)a1 + 4LL) )
    {
      if ( (_DWORD)v12 != v15[1] )
        goto LABEL_8;
      v34 = *(_BYTE **)(v13 + 112);
      v35 = &v34[16 * *(unsigned int *)(v13 + 120)];
      if ( v34 == v35 )
      {
LABEL_38:
        v60 = v11;
        v37 = *(_DWORD *)(a2 + 192);
        v15[1] = v37;
        sub_3945B70(a1 + 8, v37, (unsigned int)v12);
        v11 = v60;
        if ( (_DWORD)v12 != *(_DWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 8 * v12 + 4) )
        {
LABEL_8:
          v16 = *(_DWORD *)(a1 + 96);
          a6 = (_BYTE *)(v12 + *(_QWORD *)(a1 + 200));
          v17 = (unsigned __int8)*a6;
          if ( v17 < v16 )
          {
            v18 = *(_QWORD *)(a1 + 88);
            v19 = (unsigned __int8)*a6;
            while ( 1 )
            {
              v20 = (_DWORD *)(v18 + 12LL * v19);
              if ( (_DWORD)v12 == *v20 )
                break;
              v19 += 256;
              if ( v16 <= v19 )
                goto LABEL_24;
            }
            v6 = v16;
            v21 = (_DWORD *)(v18 + 12LL * v16);
            if ( v21 != v20 )
            {
              v22 = (unsigned __int8)*a6;
              while ( 1 )
              {
                v23 = (_DWORD *)(v18 + 12LL * v22);
                if ( (_DWORD)v12 == *v23 )
                  break;
                v22 += 256;
                if ( v16 <= v22 )
                  goto LABEL_52;
              }
              if ( v21 != v23 )
              {
                v66 += v23[2];
                goto LABEL_19;
              }
LABEL_52:
              *a6 = v16;
              v47 = *(unsigned int *)(a1 + 96);
              if ( (unsigned int)v47 >= *(_DWORD *)(a1 + 100) )
              {
                v61 = v11;
                sub_16CD150(a1 + 88, (const void *)(a1 + 104), 0, 12, v16, (int)a6);
                v47 = *(unsigned int *)(a1 + 96);
                v11 = v61;
              }
              v48 = *(_QWORD *)(a1 + 88) + 12 * v47;
              *(_QWORD *)v48 = v12 | 0xFFFFFFFF00000000LL;
              *(_DWORD *)(v48 + 8) = 0;
              v49 = *(_QWORD *)(a1 + 200);
              v6 = (unsigned int)(*(_DWORD *)(a1 + 96) + 1);
              v18 = *(_QWORD *)(a1 + 88);
              *(_DWORD *)(a1 + 96) = v6;
              v17 = *(unsigned __int8 *)(v49 + v12);
              v16 = v6;
              v66 += *(_DWORD *)(v18 + 12 * v6 - 4);
              if ( (unsigned int)v6 > v17 )
              {
LABEL_19:
                while ( 1 )
                {
                  v24 = (_DWORD *)(v18 + 12LL * v17);
                  if ( (_DWORD)v12 == *v24 )
                    break;
                  v17 += 256;
                  if ( v16 <= v17 )
                    goto LABEL_24;
                }
                if ( v24 != (_DWORD *)(v18 + 12 * v6) )
                {
                  v25 = v18 + 12 * v6 - 12;
                  if ( v24 != (_DWORD *)v25 )
                  {
                    *(_QWORD *)v24 = *(_QWORD *)v25;
                    v24[2] = *(_DWORD *)(v25 + 8);
                    *(_BYTE *)(*(_QWORD *)(a1 + 200)
                             + *(unsigned int *)(*(_QWORD *)(a1 + 88) + 12LL * *(unsigned int *)(a1 + 96) - 12)) = -85 * (((__int64)v24 - *(_QWORD *)(a1 + 88)) >> 2);
                    v16 = *(_DWORD *)(a1 + 96);
                  }
                  *(_DWORD *)(a1 + 96) = v16 - 1;
                }
              }
            }
          }
          goto LABEL_24;
        }
      }
      else
      {
        v36 = 0;
        while ( (*v34 & 6) != 0 || (unsigned int)++v36 <= 3 )
        {
          v34 += 16;
          if ( v35 == v34 )
            goto LABEL_38;
        }
      }
    }
    else if ( (_DWORD)v12 != v15[1] )
    {
      goto LABEL_8;
    }
    v38 = *(_DWORD *)(a1 + 96);
    v39 = (_BYTE *)(v12 + *(_QWORD *)(a1 + 200));
    v40 = (unsigned __int8)*v39;
    if ( v40 >= v38 )
      goto LABEL_56;
    v41 = *(_QWORD *)(a1 + 88);
    while ( 1 )
    {
      v42 = (_DWORD *)(v41 + 12LL * v40);
      if ( (_DWORD)v12 == *v42 )
        break;
      v40 += 256;
      if ( v38 <= v40 )
        goto LABEL_56;
    }
    if ( v42 == (_DWORD *)(v41 + 12LL * v38) )
    {
LABEL_56:
      *v39 = v38;
      v50 = *(unsigned int *)(a1 + 96);
      if ( (unsigned int)v50 >= *(_DWORD *)(a1 + 100) )
      {
        v62 = v11;
        sub_16CD150(a1 + 88, (const void *)(a1 + 104), 0, 12, v6, (int)a6);
        v50 = *(unsigned int *)(a1 + 96);
        v11 = v62;
      }
      v51 = *(_QWORD *)(a1 + 88) + 12 * v50;
      *(_QWORD *)v51 = v12 | 0xFFFFFFFF00000000LL;
      *(_DWORD *)(v51 + 8) = 0;
      v52 = (unsigned int)(*(_DWORD *)(a1 + 96) + 1);
      v53 = 3 * v52;
      *(_DWORD *)(a1 + 96) = v52;
      v41 = *(_QWORD *)(a1 + 88);
      v42 = (_DWORD *)(v41 + 4 * v53 - 12);
    }
    if ( v42[1] == -1 )
    {
      v43 = (_BYTE *)(*(_QWORD *)(a1 + 200) + v12);
      v44 = *(_DWORD *)(a1 + 96);
      v45 = (unsigned __int8)*v43;
      if ( v45 >= v44 )
        goto LABEL_62;
      while ( 1 )
      {
        v46 = (_DWORD *)(v41 + 12LL * v45);
        if ( v14 == *v46 )
          break;
        v45 += 256;
        if ( v44 <= v45 )
          goto LABEL_62;
      }
      if ( v46 == (_DWORD *)(v41 + 12LL * v44) )
      {
LABEL_62:
        *v43 = v44;
        v57 = *(unsigned int *)(a1 + 96);
        if ( (unsigned int)v57 >= *(_DWORD *)(a1 + 100) )
        {
          v63 = v11;
          sub_16CD150(a1 + 88, (const void *)(a1 + 104), 0, 12, v6, (int)a6);
          v57 = *(unsigned int *)(a1 + 96);
          v11 = v63;
        }
        v58 = *(_QWORD *)(a1 + 88) + 12 * v57;
        *(_QWORD *)v58 = v14 | 0xFFFFFFFF00000000LL;
        *(_DWORD *)(v58 + 8) = 0;
        v59 = (unsigned int)(*(_DWORD *)(a1 + 96) + 1);
        *(_DWORD *)(a1 + 96) = v59;
        v46 = (_DWORD *)(*(_QWORD *)(a1 + 88) + 12 * v59 - 12);
      }
      v46[1] = *(_DWORD *)(a2 + 192);
    }
LABEL_24:
    v9 += 2;
  }
  while ( v10 != v9 );
  v6 = a2;
LABEL_26:
  v26 = *(unsigned int *)(v6 + 192);
  v27 = *(_DWORD *)(a1 + 96);
  v67 = v26 | 0xFFFFFFFF00000000LL;
  v28 = v26;
  v29 = (_BYTE *)(*(_QWORD *)(a1 + 200) + v26);
  v30 = (unsigned __int8)*v29;
  if ( v30 >= v27 )
    goto LABEL_59;
  v31 = *(_QWORD *)(a1 + 88);
  while ( 1 )
  {
    v32 = (unsigned int *)(v31 + 12LL * v30);
    if ( v28 == *v32 )
      break;
    v30 += 256;
    if ( v27 <= v30 )
      goto LABEL_59;
  }
  if ( v32 == (unsigned int *)(v31 + 12LL * v27) )
  {
LABEL_59:
    *v29 = v27;
    v54 = *(unsigned int *)(a1 + 96);
    if ( (unsigned int)v54 >= *(_DWORD *)(a1 + 100) )
    {
      sub_16CD150(a1 + 88, (const void *)(a1 + 104), 0, 12, (int)v29, (int)a6);
      v54 = *(unsigned int *)(a1 + 96);
    }
    v55 = *(_QWORD *)(a1 + 88) + 12 * v54;
    *(_QWORD *)v55 = v67;
    *(_DWORD *)(v55 + 8) = 0;
    v56 = (unsigned int)(*(_DWORD *)(a1 + 96) + 1);
    *(_DWORD *)(a1 + 96) = v56;
    v32 = (unsigned int *)(*(_QWORD *)(a1 + 88) + 12 * v56 - 12);
  }
  v32[1] = -1;
  *v32 = v65;
  v32[2] = v66;
  return v66;
}
