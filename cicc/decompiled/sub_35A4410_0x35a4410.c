// Function: sub_35A4410
// Address: 0x35a4410
//
__int64 __fastcall sub_35A4410(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  __int64 v4; // rax
  unsigned int v5; // r14d
  int v6; // r15d
  unsigned __int64 v7; // rax
  unsigned int v8; // r13d
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 result; // rax
  __int64 v12; // rdi
  _QWORD *v13; // rax
  unsigned int v14; // r13d
  int v15; // edi
  __int64 v16; // rsi
  __int64 v17; // r8
  int v18; // edi
  unsigned int v19; // ecx
  __int64 v20; // rdx
  _QWORD *v21; // rax
  __int64 v22; // r13
  __int64 v23; // rax
  __int64 v24; // r13
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // r14
  __int64 v28; // r15
  unsigned int v29; // eax
  __int64 v30; // r9
  _BYTE *v31; // rsi
  __int64 v32; // rcx
  __int64 v33; // rax
  int v34; // edx
  __int64 *v35; // rax
  unsigned __int64 v36; // rdx
  __int64 v37; // rax
  __int64 *v38; // rdi
  __int64 *v39; // r14
  __int64 *v40; // r15
  __int64 v41; // r12
  __int64 v42; // rbx
  __int64 v43; // rdi
  _QWORD *v44; // rax
  __int64 v45; // rax
  _QWORD *v46; // rax
  __int64 v47; // rdx
  _QWORD *v48; // rdx
  unsigned __int64 v49; // r8
  __int64 *v50; // rax
  _QWORD *v51; // rax
  __int64 v52; // rdx
  _QWORD *v53; // rdx
  __int64 v54; // [rsp+8h] [rbp-B8h]
  unsigned __int64 v55; // [rsp+8h] [rbp-B8h]
  unsigned __int64 v56; // [rsp+18h] [rbp-A8h]
  __int64 i; // [rsp+20h] [rbp-A0h]
  __int64 v58; // [rsp+38h] [rbp-88h] BYREF
  __int64 *v59; // [rsp+40h] [rbp-80h] BYREF
  __int64 v60; // [rsp+48h] [rbp-78h]
  _BYTE v61[112]; // [rsp+50h] [rbp-70h] BYREF

  v2 = a2;
  v3 = a1;
  if ( *(_WORD *)(a2 + 68) && *(_WORD *)(a2 + 68) != 68 )
  {
    result = sub_3598E30(a1, a2);
    v14 = result;
    if ( (_DWORD)result != -1 )
    {
      v15 = *(_DWORD *)(a1 + 184);
      v16 = *(_QWORD *)(a2 + 24);
      v17 = *(_QWORD *)(v3 + 168);
      if ( v15 )
      {
        v18 = v15 - 1;
        v19 = v18 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v20 = *(_QWORD *)(v17 + 80LL * v19);
        if ( v16 == v20 )
        {
LABEL_13:
          v58 = *(_QWORD *)(v2 + 24);
          if ( (unsigned __int8)sub_359BED0(v3 + 160, &v58, &v59) )
          {
            v21 = v59 + 1;
          }
          else
          {
            v51 = sub_35A4380(v3 + 160, &v58, v59);
            v52 = v58;
            v51[9] = 0;
            *v51 = v52;
            v53 = v51 + 3;
            v21 = v51 + 1;
            *((_OWORD *)v21 + 1) = 0;
            *((_OWORD *)v21 + 2) = 0;
            *((_OWORD *)v21 + 3) = 0;
            *v21 = v53;
            v21[1] = 0x600000000LL;
          }
          result = *(_QWORD *)(*v21 + 8LL * (v14 >> 6)) & (1LL << v14);
          if ( !result )
          {
            v22 = *(_QWORD *)(v2 + 32);
            v23 = v22 + 40LL * (unsigned int)sub_2E88FE0(v2);
            v24 = *(_QWORD *)(v2 + 32);
            for ( i = v23; v24 != i; v24 += 40 )
            {
              v25 = *(_QWORD *)(v3 + 24);
              v59 = (__int64 *)v61;
              v60 = 0x400000000LL;
              v26 = *(unsigned int *)(v24 + 8);
              if ( (int)v26 < 0 )
                v27 = *(_QWORD *)(*(_QWORD *)(v25 + 56) + 16 * (v26 & 0x7FFFFFFF) + 8);
              else
                v27 = *(_QWORD *)(*(_QWORD *)(v25 + 304) + 8 * v26);
              if ( v27 )
              {
                if ( (*(_BYTE *)(v27 + 3) & 0x10) != 0 )
                {
                  while ( 1 )
                  {
                    v27 = *(_QWORD *)(v27 + 32);
                    if ( !v27 )
                      break;
                    if ( (*(_BYTE *)(v27 + 3) & 0x10) == 0 )
                      goto LABEL_21;
                  }
                }
                else
                {
LABEL_21:
                  v28 = *(_QWORD *)(v27 + 16);
LABEL_22:
                  v29 = sub_35A2540(v3, *(_DWORD *)(*(_QWORD *)(v28 + 32) + 8LL), *(_QWORD *)(v2 + 24));
                  v31 = (_BYTE *)HIDWORD(v60);
                  v32 = v29;
                  v33 = (unsigned int)v60;
                  v34 = v60;
                  if ( (unsigned int)v60 >= (unsigned __int64)HIDWORD(v60) )
                  {
                    v36 = (unsigned int)v60 + 1LL;
                    v49 = v32 | v56 & 0xFFFFFFFF00000000LL;
                    v56 = v49;
                    if ( HIDWORD(v60) < v36 )
                    {
                      v31 = v61;
                      v55 = v49;
                      sub_C8D5F0((__int64)&v59, v61, v36, 0x10u, v49, v30);
                      v33 = (unsigned int)v60;
                      v49 = v55;
                    }
                    v50 = &v59[2 * v33];
                    *v50 = v28;
                    v50[1] = v49;
                    LODWORD(v60) = v60 + 1;
                  }
                  else
                  {
                    v35 = &v59[2 * (unsigned int)v60];
                    if ( v35 )
                    {
                      *v35 = v28;
                      *((_DWORD *)v35 + 2) = v32;
                      v34 = v60;
                    }
                    v36 = (unsigned int)(v34 + 1);
                    LODWORD(v60) = v36;
                  }
                  v37 = *(_QWORD *)(v27 + 16);
                  while ( 1 )
                  {
                    v27 = *(_QWORD *)(v27 + 32);
                    if ( !v27 )
                      break;
                    if ( (*(_BYTE *)(v27 + 3) & 0x10) == 0 )
                    {
                      v28 = *(_QWORD *)(v27 + 16);
                      if ( v37 != v28 )
                        goto LABEL_22;
                    }
                  }
                  v38 = v59;
                  v39 = &v59[2 * (unsigned int)v60];
                  if ( v39 != v59 )
                  {
                    v54 = v2;
                    v40 = v59;
                    v41 = v3;
                    do
                    {
                      v42 = *v40;
                      v40 += 2;
                      v43 = *(_QWORD *)(**(_QWORD **)(v41 + 24) + 16LL);
                      v44 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, _BYTE *, unsigned __int64, __int64))(*(_QWORD *)v43 + 200LL))(
                                        v43,
                                        v31,
                                        v36,
                                        v32);
                      v31 = (_BYTE *)*(unsigned int *)(v24 + 8);
                      sub_2E8A790(v42, (int)v31, *((_DWORD *)v40 - 2), 0, v44);
                    }
                    while ( v39 != v40 );
                    v3 = v41;
                    v38 = v59;
                    v2 = v54;
                  }
                  if ( v38 != (__int64 *)v61 )
                    _libc_free((unsigned __int64)v38);
                }
              }
            }
            v45 = *(_QWORD *)(v3 + 40);
            if ( v45 )
              sub_2FAD510(*(_QWORD *)(v45 + 32), v2);
            return sub_2E88E20(v2);
          }
        }
        else
        {
          result = 1;
          while ( v20 != -4096 )
          {
            v19 = v18 & (result + v19);
            v20 = *(_QWORD *)(v17 + 80LL * v19);
            if ( v16 == v20 )
              goto LABEL_13;
            result = (unsigned int)(result + 1);
          }
        }
      }
    }
  }
  else
  {
    v4 = *(_QWORD *)(a2 + 32);
    v5 = *(_DWORD *)(v4 + 128);
    v6 = *(_DWORD *)(v4 + 8);
    v7 = sub_2EBEE90(*(_QWORD *)(a1 + 24), v5);
    v8 = sub_3598E30(a1, v7);
    if ( v8 != -1 )
    {
      v58 = *(_QWORD *)(a2 + 24);
      v12 = a1 + 192;
      if ( (unsigned __int8)sub_359BED0(v3 + 192, &v58, &v59) )
      {
        v13 = v59 + 1;
      }
      else
      {
        v46 = sub_35A4380(v12, &v58, v59);
        v47 = v58;
        v46[9] = 0;
        *v46 = v47;
        v48 = v46 + 3;
        v13 = v46 + 1;
        *v13 = v48;
        v13[1] = 0x600000000LL;
        *((_OWORD *)v13 + 1) = 0;
        *((_OWORD *)v13 + 2) = 0;
        *((_OWORD *)v13 + 3) = 0;
      }
      if ( (*(_QWORD *)(*v13 + 8LL * (v8 >> 6)) & (1LL << v8)) == 0 )
        v5 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 48LL);
    }
    sub_2EBE4E0(
      *(_QWORD *)(v3 + 24),
      v5,
      *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v3 + 24) + 56LL) + 16LL * (v6 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL);
    sub_2EBECB0(*(_QWORD **)(v3 + 24), v6, v5);
    sub_2EAB0C0(*(_QWORD *)(a2 + 32), v6);
    result = *(unsigned int *)(v3 + 488);
    if ( result + 1 > (unsigned __int64)*(unsigned int *)(v3 + 492) )
    {
      sub_C8D5F0(v3 + 480, (const void *)(v3 + 496), result + 1, 8u, v9, v10);
      result = *(unsigned int *)(v3 + 488);
    }
    *(_QWORD *)(*(_QWORD *)(v3 + 480) + 8 * result) = a2;
    ++*(_DWORD *)(v3 + 488);
  }
  return result;
}
