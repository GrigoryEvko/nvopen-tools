// Function: sub_2367B50
// Address: 0x2367b50
//
__int64 __fastcall sub_2367B50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  _QWORD *v11; // rdx
  unsigned int v12; // r13d
  __int64 v13; // rcx
  __int64 v14; // rdx
  int v15; // eax
  __int64 v16; // rdx
  unsigned int v17; // eax
  __int64 v18; // rdx
  __int64 result; // rax
  __int64 v20; // rdx
  unsigned int v21; // r13d
  __int64 v22; // rdi
  __int64 v23; // rax
  _QWORD *v24; // rax
  __int64 v25; // rcx
  _QWORD *v26; // rcx
  __int64 v27; // r14
  __int64 v28; // rax
  __int64 v29; // r14
  unsigned __int64 v30; // rdi
  unsigned __int64 *v31; // r13
  unsigned __int64 *v32; // r15
  int v33; // eax
  __int64 v34; // rdx
  size_t v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 *v39; // r14
  __int64 v40; // r13
  int v41; // eax
  __int64 *v42; // r15
  __int64 v43; // rdx
  __int64 v44; // rdx
  int v45; // eax
  int v46; // edx
  __int64 v47; // r15
  __int64 v48; // rcx
  unsigned __int64 v49; // r13
  __int64 v50; // rax
  unsigned __int64 v51; // r15
  unsigned __int64 *v52; // r14
  __int64 v53; // [rsp+8h] [rbp-48h]
  __int64 v54; // [rsp+10h] [rbp-40h]
  __int64 v55; // [rsp+10h] [rbp-40h]
  __int64 v56; // [rsp+18h] [rbp-38h]
  int v57; // [rsp+18h] [rbp-38h]
  int v58; // [rsp+18h] [rbp-38h]
  unsigned __int64 v59; // [rsp+18h] [rbp-38h]

  *(_QWORD *)a1 = *(_QWORD *)a2;
  *(_QWORD *)(a1 + 8) = *(_QWORD *)(a2 + 8);
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 24);
  *(_QWORD *)(a1 + 32) = *(_QWORD *)(a2 + 32);
  *(_QWORD *)(a1 + 40) = *(_QWORD *)(a2 + 40);
  *(_QWORD *)(a1 + 48) = *(_QWORD *)(a2 + 48);
  *(_BYTE *)(a1 + 56) = *(_BYTE *)(a2 + 56);
  v7 = *(_QWORD *)(a2 + 64);
  *(_QWORD *)(a2 + 64) = 0;
  *(_QWORD *)(a1 + 64) = v7;
  v8 = *(_QWORD *)(a2 + 72);
  *(_QWORD *)(a2 + 72) = 0;
  *(_QWORD *)(a1 + 72) = v8;
  v9 = *(_QWORD *)(a2 + 80);
  *(_QWORD *)(a2 + 80) = 0;
  *(_QWORD *)(a1 + 80) = v9;
  *(_QWORD *)(a1 + 88) = 1;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_DWORD *)(a1 + 112) = 0;
  v10 = *(_QWORD *)(a2 + 96);
  LODWORD(v9) = *(_DWORD *)(a2 + 112);
  ++*(_QWORD *)(a2 + 88);
  *(_QWORD *)(a1 + 96) = v10;
  v11 = *(_QWORD **)(a2 + 104);
  *(_DWORD *)(a1 + 112) = v9;
  *(_QWORD *)(a2 + 96) = 0;
  *(_QWORD *)(a2 + 104) = 0;
  *(_DWORD *)(a2 + 112) = 0;
  *(_QWORD *)(a1 + 104) = v11;
  *(_QWORD *)(a1 + 120) = a1 + 136;
  *(_QWORD *)(a1 + 128) = 0;
  v12 = *(_DWORD *)(a2 + 128);
  v56 = a1 + 136;
  if ( v12 )
  {
    v22 = a1 + 120;
    if ( v22 != a2 + 120 )
    {
      v23 = *(_QWORD *)(a2 + 120);
      v11 = (_QWORD *)(a2 + 136);
      if ( v23 == a2 + 136 )
      {
        sub_2358E60(v22, v12, (__int64)v11, a4, a5, a6);
        v11 = *(_QWORD **)(a1 + 120);
        v24 = *(_QWORD **)(a2 + 120);
        v25 = 4LL * *(unsigned int *)(a2 + 128);
        if ( v25 * 8 )
        {
          v26 = &v11[v25];
          do
          {
            if ( v11 )
            {
              *v11 = *v24;
              v11[1] = v24[1];
              v11[2] = v24[2];
              v11[3] = v24[3];
              v24[3] = 0;
              v24[2] = 0;
              v24[1] = 0;
            }
            v11 += 4;
            v24 += 4;
          }
          while ( v11 != v26 );
          v27 = *(unsigned int *)(a2 + 128);
          v28 = *(_QWORD *)(a2 + 120);
          *(_DWORD *)(a1 + 128) = v12;
          v54 = v28;
          v29 = v28 + 32 * v27;
          if ( v29 != v28 )
          {
            do
            {
              v30 = *(_QWORD *)(v29 - 24);
              v31 = *(unsigned __int64 **)(v29 - 16);
              v29 -= 32;
              v32 = (unsigned __int64 *)v30;
              if ( v31 != (unsigned __int64 *)v30 )
              {
                do
                {
                  v11 = v32 + 2;
                  if ( (unsigned __int64 *)*v32 != v32 + 2 )
                    _libc_free(*v32);
                  v32 += 21;
                }
                while ( v31 != v32 );
                v30 = *(_QWORD *)(v29 + 8);
              }
              if ( v30 )
                j_j___libc_free_0(v30);
            }
            while ( v29 != v54 );
          }
        }
        else
        {
          *(_DWORD *)(a1 + 128) = v12;
        }
        *(_DWORD *)(a2 + 128) = 0;
      }
      else
      {
        *(_QWORD *)(a1 + 120) = v23;
        v33 = *(_DWORD *)(a2 + 132);
        *(_DWORD *)(a1 + 128) = v12;
        *(_DWORD *)(a1 + 132) = v33;
        *(_QWORD *)(a2 + 120) = v11;
        *(_QWORD *)(a2 + 128) = 0;
      }
    }
  }
  *(_QWORD *)(a1 + 136) = a1 + 152;
  *(_QWORD *)(a1 + 144) = 0x800000000LL;
  v13 = *(unsigned int *)(a2 + 144);
  if ( (_DWORD)v13 )
    sub_2367460(v56, a2 + 136, (__int64)v11, v13, a5, a6);
  *(_QWORD *)(a1 + 5536) = 0;
  *(_QWORD *)(a1 + 5544) = 0;
  *(_DWORD *)(a1 + 5552) = 0;
  v14 = *(_QWORD *)(a2 + 5536);
  v15 = *(_DWORD *)(a2 + 5552);
  ++*(_QWORD *)(a2 + 5528);
  *(_QWORD *)(a1 + 5536) = v14;
  v16 = *(_QWORD *)(a2 + 5544);
  *(_DWORD *)(a1 + 5552) = v15;
  *(_QWORD *)(a2 + 5536) = 0;
  *(_QWORD *)(a2 + 5544) = 0;
  *(_DWORD *)(a2 + 5552) = 0;
  *(_QWORD *)(a1 + 5528) = 1;
  *(_QWORD *)(a1 + 5544) = v16;
  *(_QWORD *)(a1 + 5560) = a1 + 5576;
  *(_QWORD *)(a1 + 5568) = 0;
  v17 = *(_DWORD *)(a2 + 5568);
  if ( v17 && a1 + 5560 != a2 + 5560 )
  {
    v36 = *(_QWORD *)(a2 + 5560);
    v37 = a2 + 5576;
    if ( v36 == a2 + 5576 )
    {
      v57 = *(_DWORD *)(a2 + 5568);
      sub_2367990(a1 + 5560, v17, v37, v36, a5, a6);
      v39 = *(__int64 **)(a2 + 5560);
      v40 = *(_QWORD *)(a1 + 5560);
      v41 = v57;
      v42 = &v39[675 * *(unsigned int *)(a2 + 5568)];
      if ( v39 == v42 )
      {
        *(_DWORD *)(a1 + 5568) = v57;
      }
      else
      {
        do
        {
          if ( v40 )
          {
            v43 = *v39;
            *(_DWORD *)(v40 + 16) = 0;
            *(_DWORD *)(v40 + 20) = 8;
            *(_QWORD *)v40 = v43;
            *(_QWORD *)(v40 + 8) = v40 + 24;
            v44 = *((unsigned int *)v39 + 4);
            if ( (_DWORD)v44 )
            {
              v58 = v41;
              sub_2367460(v40 + 8, (__int64)(v39 + 1), v44, v38, a5, a6);
              v41 = v58;
            }
          }
          v39 += 675;
          v40 += 5400;
        }
        while ( v42 != v39 );
        v47 = *(unsigned int *)(a2 + 5568);
        v48 = *(_QWORD *)(a2 + 5560);
        *(_DWORD *)(a1 + 5568) = v41;
        v53 = v48;
        v55 = v48 + 5400 * v47;
        if ( v55 != v48 )
        {
          do
          {
            v55 -= 5400;
            v59 = *(_QWORD *)(v55 + 8);
            v49 = v59 + 672LL * *(unsigned int *)(v55 + 16);
            if ( v59 != v49 )
            {
              do
              {
                v50 = *(unsigned int *)(v49 - 648);
                v51 = *(_QWORD *)(v49 - 656);
                v49 -= 672LL;
                v50 *= 160;
                v52 = (unsigned __int64 *)(v51 + v50);
                if ( v51 != v51 + v50 )
                {
                  do
                  {
                    v52 -= 20;
                    if ( (unsigned __int64 *)*v52 != v52 + 2 )
                      _libc_free(*v52);
                  }
                  while ( (unsigned __int64 *)v51 != v52 );
                  v51 = *(_QWORD *)(v49 + 16);
                }
                if ( v51 != v49 + 32 )
                  _libc_free(v51);
              }
              while ( v59 != v49 );
              v59 = *(_QWORD *)(v55 + 8);
            }
            if ( v59 != v55 + 24 )
              _libc_free(v59);
          }
          while ( v55 != v53 );
        }
      }
      *(_DWORD *)(a2 + 5568) = 0;
    }
    else
    {
      *(_DWORD *)(a1 + 5568) = v17;
      v45 = *(_DWORD *)(a2 + 5572);
      *(_QWORD *)(a1 + 5560) = v36;
      *(_DWORD *)(a1 + 5572) = v45;
      *(_QWORD *)(a2 + 5560) = v37;
      *(_QWORD *)(a2 + 5568) = 0;
    }
  }
  *(_QWORD *)(a1 + 5584) = 0;
  *(_QWORD *)(a1 + 5592) = 0;
  *(_DWORD *)(a1 + 5600) = 0;
  v18 = *(_QWORD *)(a2 + 5584);
  result = *(unsigned int *)(a2 + 5600);
  ++*(_QWORD *)(a2 + 5576);
  *(_QWORD *)(a1 + 5584) = v18;
  v20 = *(_QWORD *)(a2 + 5592);
  *(_QWORD *)(a2 + 5584) = 0;
  *(_QWORD *)(a2 + 5592) = 0;
  *(_DWORD *)(a2 + 5600) = 0;
  *(_QWORD *)(a1 + 5576) = 1;
  *(_QWORD *)(a1 + 5592) = v20;
  *(_DWORD *)(a1 + 5600) = result;
  *(_QWORD *)(a1 + 5608) = a1 + 5624;
  *(_QWORD *)(a1 + 5616) = 0;
  v21 = *(_DWORD *)(a2 + 5616);
  if ( v21 )
  {
    result = a2 + 5608;
    if ( a1 + 5608 != a2 + 5608 )
    {
      v34 = *(_QWORD *)(a2 + 5608);
      result = a2 + 5624;
      if ( v34 == a2 + 5624 )
      {
        result = sub_C8D5F0(a1 + 5608, (const void *)(a1 + 5624), v21, 0x10u, a5, a6);
        v35 = 16LL * *(unsigned int *)(a2 + 5616);
        if ( v35 )
          result = (__int64)memcpy(*(void **)(a1 + 5608), *(const void **)(a2 + 5608), v35);
        *(_DWORD *)(a1 + 5616) = v21;
        *(_DWORD *)(a2 + 5616) = 0;
      }
      else
      {
        *(_QWORD *)(a1 + 5608) = v34;
        v46 = *(_DWORD *)(a2 + 5620);
        *(_DWORD *)(a1 + 5616) = v21;
        *(_DWORD *)(a1 + 5620) = v46;
        *(_QWORD *)(a2 + 5608) = result;
        *(_QWORD *)(a2 + 5616) = 0;
      }
    }
  }
  return result;
}
