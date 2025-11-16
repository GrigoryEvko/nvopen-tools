// Function: sub_1ED9360
// Address: 0x1ed9360
//
__int64 __fastcall sub_1ED9360(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // r13
  __int64 result; // rax
  __int64 *v9; // r12
  __int64 v10; // r15
  unsigned __int64 v11; // rdx
  __int64 v12; // r13
  __int64 *v13; // rax
  __int64 v14; // r14
  __int64 v15; // r10
  __int64 *v16; // rax
  __int64 v17; // r10
  __int64 v18; // rsi
  __int64 v19; // r10
  __int64 v20; // r14
  __int64 v21; // r13
  unsigned __int64 v22; // r12
  __int64 v23; // r15
  __int64 v24; // rbx
  __int64 *v25; // rdi
  __int64 *v26; // rdx
  unsigned int v27; // eax
  __int64 *v28; // rsi
  __int64 v29; // r10
  unsigned __int64 v30; // rsi
  __int64 v31; // rax
  __int64 *v32; // rsi
  unsigned int v33; // edi
  __int64 *v34; // rcx
  int v35; // r14d
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // [rsp+8h] [rbp-88h]
  __int64 **v39; // [rsp+10h] [rbp-80h]
  __int64 v40; // [rsp+18h] [rbp-78h]
  __int64 v41; // [rsp+18h] [rbp-78h]
  const void *v42; // [rsp+20h] [rbp-70h]
  unsigned __int64 v43; // [rsp+28h] [rbp-68h]
  __int64 v44; // [rsp+28h] [rbp-68h]
  __int64 v46; // [rsp+38h] [rbp-58h]
  unsigned __int64 v47; // [rsp+38h] [rbp-58h]
  __int64 v48; // [rsp+38h] [rbp-58h]
  unsigned __int64 v49; // [rsp+38h] [rbp-58h]
  unsigned __int64 v50; // [rsp+38h] [rbp-58h]
  __int64 v52; // [rsp+48h] [rbp-48h]
  __int64 v53; // [rsp+50h] [rbp-40h] BYREF
  __int64 v54; // [rsp+58h] [rbp-38h] BYREF

  v6 = *a1;
  result = *(unsigned int *)(*a1 + 72);
  if ( (_DWORD)result )
  {
    v9 = a1;
    v10 = 0;
    v52 = 8LL * (unsigned int)(result - 1);
    v42 = (const void *)(a3 + 16);
    while ( 1 )
    {
      v14 = *(_QWORD *)(*(_QWORD *)(v6 + 64) + v10);
      result = v9[14] + 5 * v10;
      v15 = *(_QWORD *)(v14 + 8);
      if ( *(_DWORD *)result )
      {
        if ( *(_DWORD *)result == 1 )
        {
          v11 = v15 & 0xFFFFFFFFFFFFFFF8LL;
          goto LABEL_5;
        }
      }
      else if ( *(_BYTE *)(result + 32) && *(_BYTE *)(result + 33) )
      {
        if ( !a4 )
        {
          v50 = v15 & 0xFFFFFFFFFFFFFFF8LL;
          sub_1DB4670(v6, v14);
          *(_QWORD *)(v14 + 8) = 0;
          v11 = v50;
          goto LABEL_5;
        }
        v43 = v15 & 0xFFFFFFFFFFFFFFF8LL;
        v46 = *(_QWORD *)(v14 + 8);
        v16 = (__int64 *)sub_1DB3C70((__int64 *)v6, v46);
        v17 = v46;
        v18 = *(_QWORD *)v6 + 24LL * *(unsigned int *)(v6 + 8);
        if ( v16 != (__int64 *)v18
          && (*(_DWORD *)((*v16 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v16 >> 1) & 3) <= (*(_DWORD *)(v43 + 24)
                                                                                                 | (unsigned int)(v46 >> 1)
                                                                                                 & 3) )
        {
          v18 = (__int64)v16;
        }
        v47 = v43;
        v40 = v17;
        v44 = *(_QWORD *)(v18 + 8);
        sub_1DB4670(*v9, v14);
        v11 = v47;
        *(_QWORD *)(v14 + 8) = 0;
        if ( *(_QWORD *)(a4 + 104) )
        {
          v19 = v40;
          v39 = (__int64 **)v9;
          v20 = 0;
          v21 = 0;
          v41 = v10;
          v22 = v47;
          v23 = *(_QWORD *)(a4 + 104);
          v38 = a2;
          v24 = v19;
          v53 = 0;
          v54 = 0;
          v48 = v19 >> 1;
          while ( 1 )
          {
            while ( 1 )
            {
              v26 = (__int64 *)sub_1DB3C70((__int64 *)v23, v24);
              if ( v26 != (__int64 *)(*(_QWORD *)v23 + 24LL * *(unsigned int *)(v23 + 8)) )
                break;
LABEL_25:
              v23 = *(_QWORD *)(v23 + 104);
              if ( !v23 )
                goto LABEL_32;
            }
            v25 = v26;
            v27 = *(_DWORD *)((*v26 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v26 >> 1) & 3;
            if ( v27 > (*(_DWORD *)(v22 + 24) | (unsigned int)(v48 & 3)) )
            {
              if ( (v21 & 0xFFFFFFFFFFFFFFF8LL) != 0
                && v27 >= (*(_DWORD *)((v21 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v21 >> 1) & 3) )
              {
                v25 = &v53;
              }
              v21 = *v25;
              v53 = *v25;
              goto LABEL_25;
            }
            v28 = v26 + 1;
            if ( (v20 & 0xFFFFFFFFFFFFFFF8LL) != 0
              && (*(_DWORD *)((v20 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v20 >> 1) & 3) >= (*(_DWORD *)((v26[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v26[1] >> 1) & 3) )
            {
              v28 = &v54;
            }
            v20 = *v28;
            v23 = *(_QWORD *)(v23 + 104);
            v54 = *v28;
            if ( !v23 )
            {
LABEL_32:
              v29 = v24;
              v11 = v22;
              v10 = v41;
              v9 = (__int64 *)v39;
              a2 = v38;
              v30 = v20 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (v20 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
              {
                a5 = v44;
                if ( (*(_DWORD *)(v30 + 24) | (unsigned int)(v20 >> 1) & 3) >= (*(_DWORD *)((v44 & 0xFFFFFFFFFFFFFFF8LL)
                                                                                          + 24)
                                                                              | (unsigned int)(v44 >> 1) & 3) )
                  v20 = v44;
                v44 = v20;
              }
              if ( (v21 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
              {
                a6 = v44;
                if ( (*(_DWORD *)((v21 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v21 >> 1) & 3) >= (*(_DWORD *)((v44 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v44 >> 1) & 3) )
                  v21 = v44;
                v44 = v21;
              }
              if ( v30 )
              {
                v49 = v11;
                v31 = sub_1DB3C70(*v39, v29);
                v11 = v49;
                if ( v31 != **v39 )
                  *(_QWORD *)(v31 - 16) = v44;
              }
              break;
            }
          }
        }
LABEL_5:
        if ( !v11 )
          BUG();
        v12 = *(_QWORD *)(v11 + 16);
        if ( **(_WORD **)(v12 + 16) == 15 )
        {
          v35 = *(_DWORD *)(*(_QWORD *)(v12 + 32) + 48LL);
          if ( v35 < 0 )
          {
            v36 = v9[4];
            if ( v35 != *(_DWORD *)(v36 + 12) && v35 != *(_DWORD *)(v36 + 8) )
            {
              v37 = *(unsigned int *)(a3 + 8);
              if ( (unsigned int)v37 >= *(_DWORD *)(a3 + 12) )
              {
                sub_16CD150(a3, v42, 0, 4, a5, a6);
                v37 = *(unsigned int *)(a3 + 8);
              }
              *(_DWORD *)(*(_QWORD *)a3 + 4 * v37) = v35;
              ++*(_DWORD *)(a3 + 8);
            }
          }
        }
        v13 = *(__int64 **)(a2 + 8);
        if ( *(__int64 **)(a2 + 16) == v13 )
        {
          v32 = &v13[*(unsigned int *)(a2 + 28)];
          v33 = *(_DWORD *)(a2 + 28);
          if ( v13 == v32 )
          {
LABEL_59:
            if ( v33 >= *(_DWORD *)(a2 + 24) )
              goto LABEL_8;
            *(_DWORD *)(a2 + 28) = v33 + 1;
            *v32 = v12;
            ++*(_QWORD *)a2;
          }
          else
          {
            v34 = 0;
            while ( v12 != *v13 )
            {
              if ( *v13 == -2 )
                v34 = v13;
              if ( v32 == ++v13 )
              {
                if ( !v34 )
                  goto LABEL_59;
                *v34 = v12;
                --*(_DWORD *)(a2 + 32);
                ++*(_QWORD *)a2;
                break;
              }
            }
          }
        }
        else
        {
LABEL_8:
          sub_16CCBA0(a2, v12);
        }
        sub_1F10740(*(_QWORD *)(v9[5] + 272), v12);
        result = sub_1E16240(v12);
      }
      if ( v10 == v52 )
        return result;
      v6 = *v9;
      v10 += 8;
    }
  }
  return result;
}
