// Function: sub_14EBA00
// Address: 0x14eba00
//
__int64 __fastcall sub_14EBA00(__int64 a1)
{
  __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  _QWORD *v4; // r13
  _QWORD *v5; // r12
  __int64 v6; // rdi
  __int64 *v7; // r12
  unsigned __int64 v8; // r13
  __int64 v9; // rdi
  __int64 v10; // rax
  _QWORD *v11; // r12
  _QWORD *v12; // r13
  __int64 v13; // rdi
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // rdi
  __int64 v21; // rdi
  __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  __int64 v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // r13
  __int64 v27; // r12
  __int64 v28; // rax
  __int64 v29; // rdi
  _QWORD *v30; // r13
  _QWORD *v31; // r12
  _QWORD *v32; // r13
  _QWORD *v33; // r12
  __int64 v34; // rdi
  __int64 v35; // r8
  unsigned __int64 v36; // r14
  __int64 v37; // rdi
  __int64 v38; // r15
  __int64 v39; // r13
  volatile signed __int32 *v40; // r12
  signed __int32 v41; // edx
  signed __int32 v42; // eax
  __int64 v43; // r14
  __int64 v44; // r12
  volatile signed __int32 *v45; // r13
  signed __int32 v46; // edx
  signed __int32 v47; // eax
  __int64 result; // rax
  _QWORD *v49; // r13
  __int64 v50; // r15
  __int64 v51; // r12
  __int64 v52; // rdi
  _QWORD *v53; // rdi
  __int64 v54; // r14
  __int64 v55; // r12
  volatile signed __int32 *v56; // r15
  signed __int32 v57; // edx
  __int64 v58; // [rsp+0h] [rbp-40h]
  _QWORD *v59; // [rsp+0h] [rbp-40h]

  *(_QWORD *)a1 = off_4984BB0;
  v2 = *(_QWORD *)(a1 + 1784);
  if ( v2 )
    j_j___libc_free_0(v2, *(_QWORD *)(a1 + 1800) - v2);
  v3 = *(_QWORD *)(a1 + 1760);
  if ( v3 != a1 + 1776 )
    _libc_free(v3);
  v4 = *(_QWORD **)(a1 + 1744);
  v5 = *(_QWORD **)(a1 + 1736);
  if ( v4 != v5 )
  {
    do
    {
      if ( (_QWORD *)*v5 != v5 + 2 )
        j_j___libc_free_0(*v5, v5[2] + 1LL);
      v5 += 4;
    }
    while ( v4 != v5 );
    v5 = *(_QWORD **)(a1 + 1736);
  }
  if ( v5 )
    j_j___libc_free_0(v5, *(_QWORD *)(a1 + 1752) - (_QWORD)v5);
  j___libc_free_0(*(_QWORD *)(a1 + 1712));
  j___libc_free_0(*(_QWORD *)(a1 + 1680));
  v6 = *(_QWORD *)(a1 + 1576);
  if ( v6 )
  {
    v7 = *(__int64 **)(a1 + 1616);
    v8 = *(_QWORD *)(a1 + 1648) + 8LL;
    if ( v8 > (unsigned __int64)v7 )
    {
      do
      {
        v9 = *v7++;
        j_j___libc_free_0(v9, 512);
      }
      while ( v8 > (unsigned __int64)v7 );
      v6 = *(_QWORD *)(a1 + 1576);
    }
    j_j___libc_free_0(v6, 8LL * *(_QWORD *)(a1 + 1584));
  }
  v10 = *(unsigned int *)(a1 + 1568);
  if ( (_DWORD)v10 )
  {
    v11 = *(_QWORD **)(a1 + 1552);
    v12 = &v11[4 * v10];
    do
    {
      if ( *v11 != -8 && *v11 != -16 )
      {
        v13 = v11[1];
        if ( v13 )
          j_j___libc_free_0(v13, v11[3] - v13);
      }
      v11 += 4;
    }
    while ( v12 != v11 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 1552));
  v14 = *(_QWORD *)(a1 + 1520);
  if ( v14 )
    j_j___libc_free_0(v14, *(_QWORD *)(a1 + 1536) - v14);
  j___libc_free_0(*(_QWORD *)(a1 + 1496));
  j___libc_free_0(*(_QWORD *)(a1 + 1456));
  j___libc_free_0(*(_QWORD *)(a1 + 1424));
  v15 = *(_QWORD *)(a1 + 1392);
  if ( v15 )
    j_j___libc_free_0(v15, *(_QWORD *)(a1 + 1408) - v15);
  v16 = *(_QWORD *)(a1 + 1368);
  if ( v16 )
    j_j___libc_free_0(v16, *(_QWORD *)(a1 + 1384) - v16);
  sub_14EA760(*(_QWORD *)(a1 + 1336));
  v17 = *(_QWORD *)(a1 + 1296);
  if ( v17 )
    j_j___libc_free_0(v17, *(_QWORD *)(a1 + 1312) - v17);
  v18 = *(_QWORD *)(a1 + 1272);
  if ( v18 )
    j_j___libc_free_0(v18, *(_QWORD *)(a1 + 1288) - v18);
  v19 = *(_QWORD *)(a1 + 1248);
  if ( v19 )
    j_j___libc_free_0(v19, *(_QWORD *)(a1 + 1264) - v19);
  v20 = *(_QWORD *)(a1 + 1224);
  if ( v20 )
    j_j___libc_free_0(v20, *(_QWORD *)(a1 + 1240) - v20);
  v21 = *(_QWORD *)(a1 + 1200);
  if ( v21 )
    j_j___libc_free_0(v21, *(_QWORD *)(a1 + 1216) - v21);
  v22 = *(_QWORD *)(a1 + 1176);
  if ( v22 )
    j_j___libc_free_0(v22, *(_QWORD *)(a1 + 1192) - v22);
  v23 = *(_QWORD *)(a1 + 648);
  if ( v23 != a1 + 664 )
    _libc_free(v23);
  v24 = *(_QWORD *)(a1 + 624);
  if ( v24 )
    j_j___libc_free_0(v24, *(_QWORD *)(a1 + 640) - v24);
  if ( *(_BYTE *)(a1 + 616) )
    sub_1517350(a1 + 608);
  v25 = *(_QWORD *)(a1 + 576);
  if ( v25 )
    j_j___libc_free_0(v25, *(_QWORD *)(a1 + 592) - v25);
  v26 = *(_QWORD *)(a1 + 560);
  v27 = *(_QWORD *)(a1 + 552);
  if ( v26 != v27 )
  {
    do
    {
      v28 = *(_QWORD *)(v27 + 16);
      if ( v28 != -8 && v28 != 0 && v28 != -16 )
        sub_1649B30(v27);
      v27 += 24;
    }
    while ( v26 != v27 );
    v27 = *(_QWORD *)(a1 + 552);
  }
  if ( v27 )
    j_j___libc_free_0(v27, *(_QWORD *)(a1 + 568) - v27);
  v29 = *(_QWORD *)(a1 + 528);
  if ( v29 )
    j_j___libc_free_0(v29, *(_QWORD *)(a1 + 544) - v29);
  v30 = *(_QWORD **)(a1 + 512);
  v31 = *(_QWORD **)(a1 + 504);
  if ( v30 != v31 )
  {
    do
    {
      if ( (_QWORD *)*v31 != v31 + 2 )
        j_j___libc_free_0(*v31, v31[2] + 1LL);
      v31 += 4;
    }
    while ( v30 != v31 );
    v31 = *(_QWORD **)(a1 + 504);
  }
  if ( v31 )
    j_j___libc_free_0(v31, *(_QWORD *)(a1 + 520) - (_QWORD)v31);
  v32 = *(_QWORD **)(a1 + 488);
  v33 = *(_QWORD **)(a1 + 480);
  if ( v32 != v33 )
  {
    do
    {
      if ( (_QWORD *)*v33 != v33 + 2 )
        j_j___libc_free_0(*v33, v33[2] + 1LL);
      v33 += 4;
    }
    while ( v32 != v33 );
    v33 = *(_QWORD **)(a1 + 480);
  }
  if ( v33 )
    j_j___libc_free_0(v33, *(_QWORD *)(a1 + 496) - (_QWORD)v33);
  nullsub_555(a1);
  v34 = *(_QWORD *)(a1 + 400);
  if ( v34 != a1 + 416 )
    j_j___libc_free_0(v34, *(_QWORD *)(a1 + 416) + 1LL);
  v35 = 32LL * *(unsigned int *)(a1 + 104);
  v58 = *(_QWORD *)(a1 + 96);
  v36 = v58 + v35;
  if ( v58 != v58 + v35 )
  {
    do
    {
      v37 = *(_QWORD *)(v36 - 24);
      v38 = *(_QWORD *)(v36 - 16);
      v36 -= 32LL;
      v39 = v37;
      if ( v38 != v37 )
      {
        do
        {
          while ( 1 )
          {
            v40 = *(volatile signed __int32 **)(v39 + 8);
            if ( v40 )
            {
              if ( &_pthread_key_create )
              {
                v41 = _InterlockedExchangeAdd(v40 + 2, 0xFFFFFFFF);
              }
              else
              {
                v41 = *((_DWORD *)v40 + 2);
                *((_DWORD *)v40 + 2) = v41 - 1;
              }
              if ( v41 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v40 + 16LL))(v40);
                if ( &_pthread_key_create )
                {
                  v42 = _InterlockedExchangeAdd(v40 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v42 = *((_DWORD *)v40 + 3);
                  *((_DWORD *)v40 + 3) = v42 - 1;
                }
                if ( v42 == 1 )
                  break;
              }
            }
            v39 += 16;
            if ( v38 == v39 )
              goto LABEL_88;
          }
          v39 += 16;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v40 + 24LL))(v40);
        }
        while ( v38 != v39 );
LABEL_88:
        v37 = *(_QWORD *)(v36 + 8);
      }
      if ( v37 )
        j_j___libc_free_0(v37, *(_QWORD *)(v36 + 24) - v37);
    }
    while ( v58 != v36 );
    v36 = *(_QWORD *)(a1 + 96);
  }
  if ( v36 != a1 + 112 )
    _libc_free(v36);
  v43 = *(_QWORD *)(a1 + 80);
  v44 = *(_QWORD *)(a1 + 72);
  if ( v43 != v44 )
  {
    do
    {
      while ( 1 )
      {
        v45 = *(volatile signed __int32 **)(v44 + 8);
        if ( v45 )
        {
          if ( &_pthread_key_create )
          {
            v46 = _InterlockedExchangeAdd(v45 + 2, 0xFFFFFFFF);
          }
          else
          {
            v46 = *((_DWORD *)v45 + 2);
            *((_DWORD *)v45 + 2) = v46 - 1;
          }
          if ( v46 == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v45 + 16LL))(v45);
            if ( &_pthread_key_create )
            {
              v47 = _InterlockedExchangeAdd(v45 + 3, 0xFFFFFFFF);
            }
            else
            {
              v47 = *((_DWORD *)v45 + 3);
              *((_DWORD *)v45 + 3) = v47 - 1;
            }
            if ( v47 == 1 )
              break;
          }
        }
        v44 += 16;
        if ( v43 == v44 )
          goto LABEL_106;
      }
      v44 += 16;
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v45 + 24LL))(v45);
    }
    while ( v43 != v44 );
LABEL_106:
    v44 = *(_QWORD *)(a1 + 72);
  }
  if ( v44 )
    j_j___libc_free_0(v44, *(_QWORD *)(a1 + 88) - v44);
  result = *(_QWORD *)(a1 + 16);
  v49 = *(_QWORD **)(a1 + 8);
  v59 = (_QWORD *)result;
  if ( (_QWORD *)result != v49 )
  {
    do
    {
      v50 = v49[9];
      v51 = v49[8];
      if ( v50 != v51 )
      {
        do
        {
          v52 = *(_QWORD *)(v51 + 8);
          if ( v52 != v51 + 24 )
            j_j___libc_free_0(v52, *(_QWORD *)(v51 + 24) + 1LL);
          v51 += 40;
        }
        while ( v50 != v51 );
        v51 = v49[8];
      }
      if ( v51 )
        j_j___libc_free_0(v51, v49[10] - v51);
      v53 = (_QWORD *)v49[4];
      result = (__int64)(v49 + 6);
      if ( v53 != v49 + 6 )
        result = j_j___libc_free_0(v53, v49[6] + 1LL);
      v54 = v49[2];
      v55 = v49[1];
      if ( v54 != v55 )
      {
        do
        {
          while ( 1 )
          {
            v56 = *(volatile signed __int32 **)(v55 + 8);
            if ( v56 )
            {
              result = (__int64)&_pthread_key_create;
              if ( &_pthread_key_create )
              {
                v57 = _InterlockedExchangeAdd(v56 + 2, 0xFFFFFFFF);
              }
              else
              {
                v57 = *((_DWORD *)v56 + 2);
                *((_DWORD *)v56 + 2) = v57 - 1;
              }
              if ( v57 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v56 + 16LL))(v56);
                if ( &_pthread_key_create )
                {
                  result = (unsigned int)_InterlockedExchangeAdd(v56 + 3, 0xFFFFFFFF);
                }
                else
                {
                  result = *((unsigned int *)v56 + 3);
                  *((_DWORD *)v56 + 3) = result - 1;
                }
                if ( (_DWORD)result == 1 )
                  break;
              }
            }
            v55 += 16;
            if ( v54 == v55 )
              goto LABEL_130;
          }
          v55 += 16;
          result = (*(__int64 (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v56 + 24LL))(v56);
        }
        while ( v54 != v55 );
LABEL_130:
        v55 = v49[1];
      }
      if ( v55 )
        result = j_j___libc_free_0(v55, v49[3] - v55);
      v49 += 11;
    }
    while ( v59 != v49 );
    v49 = *(_QWORD **)(a1 + 8);
  }
  if ( v49 )
    return j_j___libc_free_0(v49, *(_QWORD *)(a1 + 24) - (_QWORD)v49);
  return result;
}
