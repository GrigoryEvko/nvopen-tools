// Function: sub_289BA10
// Address: 0x289ba10
//
__int64 __fastcall sub_289BA10(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int i; // ebx
  __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // r8
  __int64 v8; // r9
  int v9; // eax
  _QWORD *v10; // rax
  __int64 *v11; // r12
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // r10
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // r8
  __int64 v26; // r10
  unsigned int v27; // eax
  __int64 v28; // rbx
  unsigned __int8 *v29; // r12
  __int64 v30; // rax
  _QWORD *v31; // rax
  unsigned __int8 *v32; // rdx
  __int64 v33; // [rsp+0h] [rbp-C0h]
  __int64 v35; // [rsp+18h] [rbp-A8h]
  __int64 *v36; // [rsp+18h] [rbp-A8h]
  _BYTE *v37; // [rsp+18h] [rbp-A8h]
  __int64 v38; // [rsp+18h] [rbp-A8h]
  unsigned __int8 *v39; // [rsp+20h] [rbp-A0h] BYREF
  _BYTE *v40; // [rsp+28h] [rbp-98h] BYREF
  __int64 v41; // [rsp+30h] [rbp-90h] BYREF
  __int64 v42; // [rsp+38h] [rbp-88h] BYREF
  _QWORD *v43; // [rsp+40h] [rbp-80h] BYREF
  int v44; // [rsp+48h] [rbp-78h]
  _BYTE **v45; // [rsp+50h] [rbp-70h]
  int v46; // [rsp+58h] [rbp-68h]
  int v47; // [rsp+60h] [rbp-60h]
  int v48; // [rsp+68h] [rbp-58h]
  unsigned int v49; // [rsp+70h] [rbp-50h]
  __int64 *v50; // [rsp+78h] [rbp-48h]
  unsigned int v51; // [rsp+80h] [rbp-40h]
  __int64 *v52; // [rsp+88h] [rbp-38h]

  *(_QWORD *)(a1 + 8) = 0x2000000000LL;
  *(_QWORD *)a1 = a1 + 16;
  i = *(_DWORD *)(a3 + 8);
  if ( i )
  {
    while ( 1 )
    {
      v5 = i--;
      v6 = *(_QWORD *)(*(_QWORD *)a3 + 8 * v5 - 8);
      *(_DWORD *)(a3 + 8) = i;
      v39 = (unsigned __int8 *)v6;
      if ( *(_BYTE *)v6 > 0x1Cu )
        break;
LABEL_3:
      if ( !i )
        return a1;
    }
    if ( *(_BYTE *)v6 != 85 )
      goto LABEL_6;
    v17 = *(_QWORD *)(v6 - 32);
    if ( !v17 )
      goto LABEL_6;
    if ( !*(_BYTE *)v17
      && *(_QWORD *)(v17 + 24) == *(_QWORD *)(v6 + 80)
      && *(_DWORD *)(v17 + 36) == 233
      && *(_QWORD *)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF)) )
    {
      v40 = *(_BYTE **)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
      if ( *(_BYTE *)v6 != 85 )
        goto LABEL_6;
      v21 = *(_DWORD *)(v6 + 4) & 0x7FFFFFF;
      v37 = *(_BYTE **)(v6 + 32 * (1 - v21));
      if ( v37 )
      {
        v22 = 32 * (2 - v21);
        v23 = *(_QWORD *)(v6 + v22);
        if ( v23 )
        {
          v41 = *(_QWORD *)(v6 + v22);
          if ( *(_BYTE *)v6 != 85 )
            goto LABEL_6;
          v24 = 32 * (3LL - (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
          v25 = *(_QWORD *)(v6 + v24);
          if ( v25 )
          {
            v42 = *(_QWORD *)(v6 + v24);
            if ( *(_BYTE *)v6 != 85 )
              goto LABEL_6;
            v33 = *(_QWORD *)(v6 + 32 * (4LL - (*(_DWORD *)(v6 + 4) & 0x7FFFFFF)));
            if ( v33 )
            {
              sub_28940A0((__int64)&v43, v23, v25);
              if ( (unsigned __int8)sub_2896BA0(a2, (__int64)v40, (__int64)v43, v44) && *v40 > 0x1Cu )
                sub_9C95B0(a3, (__int64)v40);
              sub_28940A0((__int64)&v43, v42, v33);
              if ( (unsigned __int8)sub_2896BA0(a2, (__int64)v37, (__int64)v43, v44) && *v37 > 0x1Cu )
                sub_9C95B0(a3, (__int64)v37);
LABEL_59:
              v26 = i;
              v27 = *(_DWORD *)(a3 + 8);
              for ( i = v27; v26 != v27; i = v27 )
              {
                v28 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a3 + 8 * v26) + 16LL);
                if ( v28 )
                {
                  do
                  {
                    v29 = *(unsigned __int8 **)(v28 + 24);
                    if ( *v29 > 0x1Cu && v39 != v29 )
                    {
                      v30 = *(unsigned int *)(a1 + 8);
                      if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
                      {
                        v38 = v26;
                        sub_C8D5F0(a1, (const void *)(a1 + 16), v30 + 1, 8u, v7, v8);
                        v30 = *(unsigned int *)(a1 + 8);
                        v26 = v38;
                      }
                      *(_QWORD *)(*(_QWORD *)a1 + 8 * v30) = v29;
                      ++*(_DWORD *)(a1 + 8);
                    }
                    v28 = *(_QWORD *)(v28 + 8);
                  }
                  while ( v28 );
                  v27 = *(_DWORD *)(a3 + 8);
                }
                ++v26;
              }
              goto LABEL_3;
            }
          }
        }
      }
      v17 = *(_QWORD *)(v6 - 32);
    }
    if ( v17 )
    {
      if ( !*(_BYTE *)v17 && *(_QWORD *)(v17 + 24) == *(_QWORD *)(v6 + 80) && *(_DWORD *)(v17 + 36) == 234 )
      {
        if ( *(_QWORD *)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF)) )
        {
          v40 = *(_BYTE **)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
          if ( *(_BYTE *)v6 == 85 )
          {
            v18 = 32 * (1LL - (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
            v16 = *(_QWORD *)(v6 + v18);
            if ( v16 )
            {
              v41 = *(_QWORD *)(v6 + v18);
              if ( *(_BYTE *)v6 == 85 )
              {
                v19 = 32 * (2LL - (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
                v20 = *(_QWORD *)(v6 + v19);
                if ( v20 )
                {
                  v42 = *(_QWORD *)(v6 + v19);
                  v15 = v20;
LABEL_28:
                  sub_28940A0((__int64)&v43, v16, v15);
                  if ( (unsigned __int8)sub_2896BA0(a2, (__int64)v40, (__int64)v43, v44) && *v40 > 0x1Cu )
                    sub_9C95B0(a3, (__int64)v40);
                  goto LABEL_59;
                }
              }
            }
          }
        }
      }
    }
LABEL_6:
    v35 = v6;
    LODWORD(v43) = 232;
    v45 = &v40;
    v50 = &v41;
    v44 = 0;
    v46 = 1;
    v47 = 2;
    v48 = 3;
    v49 = 4;
    v51 = 5;
    v52 = &v42;
    if ( !(unsigned __int8)sub_10E25C0((__int64)&v43, v6)
      || *(_BYTE *)v35 != 85
      || (v13 = *(_QWORD *)(v35 + 32 * (v49 - (unsigned __int64)(*(_DWORD *)(v35 + 4) & 0x7FFFFFF)))) == 0
      || (*v50 = v13, *(_BYTE *)v35 != 85)
      || (v14 = *(_QWORD *)(v35 + 32 * (v51 - (unsigned __int64)(*(_DWORD *)(v35 + 4) & 0x7FFFFFF)))) == 0 )
    {
      v9 = *v39;
      if ( (unsigned __int8)v9 <= 0x1Cu
        || (_BYTE)v9 != 61 && (_BYTE)v9 != 85 && (_BYTE)v9 != 62 && (unsigned int)(v9 - 41) <= 6 )
      {
        if ( (unsigned __int8)sub_28941A0(a2 + 64, (__int64 *)&v39, &v43) )
        {
          v10 = v43 + 1;
        }
        else
        {
          v31 = sub_2895620(a2 + 64, (__int64 *)&v39, v43);
          v32 = v39;
          v31[1] = 0;
          *v31 = v32;
          *((_BYTE *)v31 + 16) = dword_5003CC8 == 0;
          v10 = v31 + 1;
        }
        v43 = (_QWORD *)*v10;
        v44 = *((_DWORD *)v10 + 2);
        if ( (v39[7] & 0x40) != 0 )
        {
          v11 = (__int64 *)*((_QWORD *)v39 - 1);
          v36 = &v11[4 * (*((_DWORD *)v39 + 1) & 0x7FFFFFF)];
        }
        else
        {
          v36 = (__int64 *)v39;
          v11 = (__int64 *)&v39[-32 * (*((_DWORD *)v39 + 1) & 0x7FFFFFF)];
        }
        while ( v36 != v11 )
        {
          if ( (unsigned __int8)sub_2896BA0(a2, *v11, (__int64)v43, v44) && *(_BYTE *)*v11 > 0x1Cu )
            sub_9C95B0(a3, *v11);
          v11 += 4;
        }
      }
      goto LABEL_59;
    }
    *v52 = v14;
    v15 = v42;
    v16 = v41;
    goto LABEL_28;
  }
  return a1;
}
