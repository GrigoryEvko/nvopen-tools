// Function: sub_35EBE30
// Address: 0x35ebe30
//
__int64 __fastcall sub_35EBE30(__int64 a1, unsigned int a2, int a3)
{
  __int64 *v5; // rax
  __int64 *v6; // r12
  __int64 *v7; // rbx
  unsigned int v8; // r8d
  __int64 *v10; // r15
  __int64 v11; // rsi
  int v12; // r9d
  __int64 *v13; // rdx
  unsigned int v14; // eax
  _QWORD *v15; // r12
  __int64 v16; // rcx
  int *v17; // r12
  _BOOL4 v18; // eax
  __int64 v19; // r9
  int v20; // edx
  int v21; // ecx
  __int64 v22; // rsi
  __int64 v23; // rdx
  unsigned __int64 v24; // rcx
  int v25; // eax
  __int64 v26; // rcx
  _DWORD *v27; // rsi
  __int64 v28; // rdx
  unsigned int v29; // ecx
  int v30; // eax
  __int64 v31; // r8
  int v32; // r9d
  __int64 *v33; // r11
  __int64 v34; // rax
  unsigned __int64 v35; // rax
  __int64 v36; // rax
  unsigned __int64 v37; // rdi
  __int64 v38; // rdx
  int v39; // eax
  int v40; // r9d
  unsigned int v41; // ecx
  __int64 v42; // r8
  int v43; // [rsp+8h] [rbp-B8h]
  __int64 v44; // [rsp+10h] [rbp-B0h]
  unsigned __int64 v45; // [rsp+18h] [rbp-A8h]
  char v46; // [rsp+27h] [rbp-99h]
  __int64 v47; // [rsp+30h] [rbp-90h]
  unsigned __int64 v48; // [rsp+48h] [rbp-78h] BYREF
  _DWORD v49[4]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v50; // [rsp+60h] [rbp-60h]
  __int64 v51; // [rsp+70h] [rbp-50h] BYREF
  __int64 v52; // [rsp+78h] [rbp-48h]
  int v53; // [rsp+80h] [rbp-40h]
  int v54; // [rsp+84h] [rbp-3Ch]
  unsigned int v55; // [rsp+88h] [rbp-38h]

  *(_DWORD *)(a1 + 6440) = a3;
  *(_DWORD *)(a1 + 6444) = a2;
  *(_DWORD *)(a1 + 280) = 0;
  sub_35EB3E0((__int64)&v51, a1, a2, a3);
  if ( *(_DWORD *)(a1 + 256) )
  {
    v5 = *(__int64 **)(a1 + 248);
    v6 = &v5[2 * *(unsigned int *)(a1 + 264)];
    if ( v5 != v6 )
    {
      while ( 1 )
      {
        v7 = v5;
        if ( *v5 != -8192 && *v5 != -4096 )
          break;
        v5 += 2;
        if ( v6 == v5 )
          return sub_C7D6A0(v52, 16LL * v55, 8);
      }
      if ( v6 != v5 )
      {
        v8 = v55;
        v10 = v6;
        v47 = a1 + 272;
        if ( !v55 )
          goto LABEL_22;
LABEL_9:
        v11 = *v7;
        v12 = 1;
        v13 = 0;
        v14 = (v8 - 1) & (((unsigned int)*v7 >> 9) ^ ((unsigned int)*v7 >> 4));
        v15 = (_QWORD *)(v52 + 16LL * v14);
        v16 = *v15;
        if ( *v7 == *v15 )
          goto LABEL_10;
        while ( v16 != -4096 )
        {
          if ( !v13 && v16 == -8192 )
            v13 = v15;
          v14 = (v8 - 1) & (v12 + v14);
          v15 = (_QWORD *)(v52 + 16LL * v14);
          v16 = *v15;
          if ( v11 == *v15 )
          {
LABEL_10:
            v17 = (int *)(v15 + 1);
            goto LABEL_11;
          }
          ++v12;
        }
        if ( !v13 )
          v13 = v15;
        ++v51;
        v30 = v53 + 1;
        if ( 4 * (v53 + 1) >= 3 * v8 )
        {
          while ( 1 )
          {
            sub_354C5D0((__int64)&v51, 2 * v8);
            if ( !v55 )
              goto LABEL_70;
            v29 = (v55 - 1) & (((unsigned int)*v7 >> 9) ^ ((unsigned int)*v7 >> 4));
            v30 = v53 + 1;
            v13 = (__int64 *)(v52 + 16LL * v29);
            v31 = *v13;
            if ( *v7 != *v13 )
              break;
LABEL_40:
            v53 = v30;
            if ( *v13 != -4096 )
              --v54;
            v34 = *v7;
            v17 = (int *)(v13 + 1);
            *((_DWORD *)v13 + 2) = 0;
            *v13 = v34;
            v11 = *v7;
LABEL_11:
            v18 = sub_35E7250(a1, v11, a2);
            v20 = *((_DWORD *)v7 + 2);
            v21 = *v17;
            v22 = *v7;
            v49[1] = v18;
            v49[2] = v20;
            v23 = *(unsigned int *)(a1 + 280);
            v49[0] = v21;
            v24 = *(unsigned int *)(a1 + 284);
            v50 = v22;
            v25 = v23;
            if ( v23 + 1 > v24 )
            {
              v35 = *(_QWORD *)(a1 + 272);
              if ( v35 > (unsigned __int64)v49 || (unsigned __int64)v49 >= v35 + 24 * v23 )
              {
                v46 = 0;
                v45 = -1;
              }
              else
              {
                v46 = 1;
                v45 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v49 - v35) >> 3);
              }
              v26 = sub_C8D7D0(v47, a1 + 288, v23 + 1, 0x18u, &v48, v19);
              v36 = *(_QWORD *)(a1 + 272);
              v37 = v36 + 24LL * *(unsigned int *)(a1 + 280);
              if ( v36 != v37 )
              {
                v38 = v26;
                do
                {
                  if ( v38 )
                  {
                    *(_DWORD *)v38 = *(_DWORD *)v36;
                    *(_DWORD *)(v38 + 4) = *(_DWORD *)(v36 + 4);
                    *(_DWORD *)(v38 + 8) = *(_DWORD *)(v36 + 8);
                    *(_QWORD *)(v38 + 16) = *(_QWORD *)(v36 + 16);
                  }
                  v36 += 24;
                  v38 += 24;
                }
                while ( v37 != v36 );
                v37 = *(_QWORD *)(a1 + 272);
              }
              v39 = v48;
              if ( v37 != a1 + 288 )
              {
                v43 = v48;
                v44 = v26;
                _libc_free(v37);
                v39 = v43;
                v26 = v44;
              }
              v23 = *(unsigned int *)(a1 + 280);
              *(_DWORD *)(a1 + 284) = v39;
              *(_QWORD *)(a1 + 272) = v26;
              v27 = v49;
              v25 = v23;
              if ( v46 )
                v27 = (_DWORD *)(v26 + 24 * v45);
            }
            else
            {
              v26 = *(_QWORD *)(a1 + 272);
              v27 = v49;
            }
            v28 = v26 + 24 * v23;
            if ( v28 )
            {
              *(_DWORD *)v28 = *v27;
              *(_DWORD *)(v28 + 4) = v27[1];
              *(_DWORD *)(v28 + 8) = v27[2];
              *(_QWORD *)(v28 + 16) = *((_QWORD *)v27 + 2);
              v25 = *(_DWORD *)(a1 + 280);
            }
            v7 += 2;
            *(_DWORD *)(a1 + 280) = v25 + 1;
            if ( v7 == v10 )
              return sub_C7D6A0(v52, 16LL * v55, 8);
            while ( *v7 == -8192 || *v7 == -4096 )
            {
              v7 += 2;
              if ( v10 == v7 )
                return sub_C7D6A0(v52, 16LL * v55, 8);
            }
            if ( v10 == v7 )
              return sub_C7D6A0(v52, 16LL * v55, 8);
            v8 = v55;
            if ( v55 )
              goto LABEL_9;
LABEL_22:
            ++v51;
          }
          v32 = 1;
          v33 = 0;
          while ( v31 != -4096 )
          {
            if ( v31 == -8192 && !v33 )
              v33 = v13;
            v29 = (v55 - 1) & (v32 + v29);
            v13 = (__int64 *)(v52 + 16LL * v29);
            v31 = *v13;
            if ( *v7 == *v13 )
              goto LABEL_40;
            ++v32;
          }
        }
        else
        {
          if ( v8 - v54 - v30 > v8 >> 3 )
            goto LABEL_40;
          sub_354C5D0((__int64)&v51, v8);
          if ( !v55 )
          {
LABEL_70:
            ++v53;
            BUG();
          }
          v40 = 1;
          v33 = 0;
          v41 = (v55 - 1) & (((unsigned int)*v7 >> 9) ^ ((unsigned int)*v7 >> 4));
          v30 = v53 + 1;
          v13 = (__int64 *)(v52 + 16LL * v41);
          v42 = *v13;
          if ( *v7 == *v13 )
            goto LABEL_40;
          while ( v42 != -4096 )
          {
            if ( v42 == -8192 && !v33 )
              v33 = v13;
            v41 = (v55 - 1) & (v40 + v41);
            v13 = (__int64 *)(v52 + 16LL * v41);
            v42 = *v13;
            if ( *v7 == *v13 )
              goto LABEL_40;
            ++v40;
          }
        }
        if ( v33 )
          v13 = v33;
        goto LABEL_40;
      }
    }
  }
  return sub_C7D6A0(v52, 16LL * v55, 8);
}
