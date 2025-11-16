// Function: sub_2E7F590
// Address: 0x2e7f590
//
__int64 __fastcall sub_2E7F590(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 i, __int64 a6)
{
  __int64 (*v6)(void); // rax
  __int64 v7; // rbx
  __int64 v8; // r15
  __int64 v9; // rdi
  __int64 v10; // rsi
  unsigned __int64 j; // rdx
  unsigned __int64 v13; // r14
  int v14; // r13d
  bool v15; // zf
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rsi
  _BYTE *v18; // r12
  _BYTE *v19; // r15
  _BYTE *v20; // rbx
  _BYTE *v21; // r15
  _DWORD *v22; // rbx
  _DWORD *v23; // r12
  _BYTE *v24; // r15
  _BYTE *v25; // rbx
  _BYTE *v26; // r15
  unsigned __int64 v27; // r15
  unsigned __int64 v28; // rax
  __int64 (__fastcall *v29)(__int64); // rax
  __int64 v30; // rax
  unsigned int v31; // r12d
  __int64 v32; // rdx
  unsigned int v33; // eax
  unsigned __int64 v34; // rdi
  __int64 v35; // [rsp+8h] [rbp-C8h]
  __int64 v36; // [rsp+10h] [rbp-C0h]
  __int64 v37; // [rsp+18h] [rbp-B8h]
  __int64 v38; // [rsp+28h] [rbp-A8h]
  _QWORD *v39; // [rsp+30h] [rbp-A0h]
  char v40; // [rsp+3Fh] [rbp-91h]
  char v41; // [rsp+3Fh] [rbp-91h]
  unsigned __int64 v42; // [rsp+40h] [rbp-90h]
  __int64 v44; // [rsp+50h] [rbp-80h]
  __int64 v45; // [rsp+58h] [rbp-78h]
  _BYTE v46[32]; // [rsp+60h] [rbp-70h] BYREF
  __int64 v47; // [rsp+80h] [rbp-50h] BYREF
  __int64 v48; // [rsp+88h] [rbp-48h]
  __int64 v49; // [rsp+90h] [rbp-40h]
  unsigned int v50; // [rsp+98h] [rbp-38h]

  v39 = 0;
  v6 = *(__int64 (**)(void))(**(_QWORD **)(a1 + 16) + 128LL);
  if ( v6 != sub_2DAC790 )
    v39 = (_QWORD *)v6();
  v47 = 0;
  v48 = 0;
  v7 = *(_QWORD *)(a1 + 328);
  v49 = 0;
  v50 = 0;
  v37 = v7;
  v35 = a1 + 320;
  if ( v7 == a1 + 320 )
  {
    v9 = 0;
    v10 = 0;
    return sub_C7D6A0(v9, v10, 4);
  }
  do
  {
    v36 = v37 + 48;
    v8 = *(_QWORD *)(v37 + 56);
    if ( v8 != v37 + 48 )
    {
      while ( 1 )
      {
        if ( *(_WORD *)(v8 + 68) == 16 )
        {
          j = *(_QWORD *)(v8 + 32);
          v13 = j + 80;
          v42 = j + 40LL * (*(_DWORD *)(v8 + 40) & 0xFFFFFF);
          if ( v42 != j + 80 )
            break;
        }
LABEL_8:
        if ( (*(_BYTE *)v8 & 4) != 0 )
        {
          v8 = *(_QWORD *)(v8 + 8);
          if ( v36 == v8 )
            goto LABEL_10;
        }
        else
        {
          while ( (*(_BYTE *)(v8 + 44) & 8) != 0 )
            v8 = *(_QWORD *)(v8 + 8);
          v8 = *(_QWORD *)(v8 + 8);
          if ( v36 == v8 )
            goto LABEL_10;
        }
      }
      v45 = v8;
LABEL_18:
      while ( *(_BYTE *)v13 )
      {
LABEL_65:
        v13 += 40LL;
        if ( v42 == v13 )
        {
LABEL_66:
          v8 = v45;
          goto LABEL_8;
        }
      }
      v14 = *(_DWORD *)(v13 + 8);
      if ( v14 )
      {
        j = *(_QWORD *)(a1 + 32);
        if ( v14 < 0 )
        {
          v44 = *(_QWORD *)(*(_QWORD *)(j + 56) + 16LL * (v14 & 0x7FFFFFFF) + 8);
        }
        else
        {
          j = *(_QWORD *)(j + 304);
          v44 = *(_QWORD *)(j + 8LL * (unsigned int)v14);
        }
        a4 = v44;
        if ( v44 )
        {
          v40 = *(_BYTE *)(v44 + 3);
          v38 = *(_QWORD *)(v44 + 32);
          v15 = (v40 & 0x10) == 0;
          v41 = v40 & 0x10;
          if ( !v15 )
          {
LABEL_24:
            v16 = *(_QWORD *)(a4 + 16);
            v17 = v16;
            for ( i = *(_DWORD *)(v16 + 44) & 4;
                  (*(_BYTE *)(v17 + 44) & 4) != 0;
                  v17 = *(_QWORD *)v17 & 0xFFFFFFFFFFFFFFF8LL )
            {
              ;
            }
            while ( 1 )
            {
              a4 = *(_QWORD *)(a4 + 32);
              if ( !a4 || (*(_BYTE *)(a4 + 3) & 0x10) == 0 )
                break;
              for ( j = *(_QWORD *)(a4 + 16); (*(_BYTE *)(j + 44) & 4) != 0; j = *(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL )
                ;
              if ( j != v17 )
                goto LABEL_39;
            }
            if ( (_DWORD)i )
            {
              do
                v16 = *(_QWORD *)v16 & 0xFFFFFFFFFFFFFFF8LL;
              while ( (*(_BYTE *)(v16 + 44) & 4) != 0 );
            }
            j = *(_QWORD *)(v16 + 32);
            v18 = (_BYTE *)(j + 40LL * (*(_DWORD *)(v16 + 40) & 0xFFFFFF));
            if ( (_BYTE *)j != v18 )
            {
              v19 = *(_BYTE **)(v16 + 32);
              while ( 1 )
              {
                v20 = v19;
                if ( sub_2DADC00(v19) )
                  break;
                v19 += 40;
                if ( v18 == v19 )
                  goto LABEL_39;
              }
              while ( v20 != v18 )
              {
                if ( v14 == *((_DWORD *)v20 + 2) )
                {
                  if ( v20 == v18 )
                    goto LABEL_39;
                  if ( v18 != v20 + 40 )
                  {
                    v24 = v20 + 40;
                    while ( 1 )
                    {
                      v25 = v24;
                      if ( sub_2DADC00(v24) )
                        break;
                      v24 += 40;
                      if ( v18 == v24 )
                        goto LABEL_62;
                    }
                    if ( v18 != v24 )
                    {
                      while ( *((_DWORD *)v25 + 2) != v14 )
                      {
                        v26 = v25 + 40;
                        if ( v25 + 40 != v18 )
                        {
                          while ( 1 )
                          {
                            v25 = v26;
                            if ( sub_2DADC00(v26) )
                              break;
                            v26 += 40;
                            if ( v18 == v26 )
                              goto LABEL_62;
                          }
                          if ( v26 != v18 )
                            continue;
                        }
                        goto LABEL_62;
                      }
                      if ( v18 != v25 )
                        goto LABEL_39;
                    }
                  }
LABEL_62:
                  if ( !v41 )
                  {
                    if ( v38 )
                    {
                      if ( (*(_BYTE *)(v38 + 3) & 0x10) == 0 )
                        BUG();
                      v44 = v38;
                    }
                    else
                    {
                      v44 = 0;
                    }
                  }
                  v27 = *(_QWORD *)(v44 + 16);
                  if ( ((*(_WORD *)(v27 + 68) - 12) & 0xFFF7) == 0
                    || (v29 = *(__int64 (__fastcall **)(__int64))(*v39 + 520LL), v29 != sub_2DCA430)
                    && (((void (__fastcall *)(_BYTE *, _QWORD *, _QWORD))v29)(v46, v39, *(_QWORD *)(v44 + 16)), v46[16]) )
                  {
                    v28 = sub_2E7F230(a1, v27, (__int64)&v47);
                    sub_2EAB510(v13, v28, HIDWORD(v28), 0);
                    goto LABEL_65;
                  }
                  v30 = *(_QWORD *)(v27 + 32);
                  v31 = 0;
                  v32 = v30 + 40LL * (*(_DWORD *)(v27 + 40) & 0xFFFFFF);
                  if ( v30 == v32 )
                  {
                    v31 = 0;
                  }
                  else
                  {
                    do
                    {
                      if ( !*(_BYTE *)v30 && (*(_BYTE *)(v30 + 3) & 0x10) != 0 && v14 == *(_DWORD *)(v30 + 8) )
                        break;
                      v30 += 40;
                      ++v31;
                    }
                    while ( v32 != v30 );
                  }
                  v33 = sub_2E8E690(v27);
                  v34 = v13;
                  v13 += 40LL;
                  sub_2EAB510(v34, v33, v31, 0);
                  if ( v42 != v13 )
                    goto LABEL_18;
                  goto LABEL_66;
                }
                v21 = v20 + 40;
                if ( v18 == v20 + 40 )
                  goto LABEL_39;
                while ( 1 )
                {
                  v20 = v21;
                  if ( sub_2DADC00(v21) )
                    break;
                  v21 += 40;
                  if ( v18 == v21 )
                    goto LABEL_39;
                }
              }
            }
            goto LABEL_39;
          }
          if ( v38 && (*(_BYTE *)(v38 + 3) & 0x10) != 0 )
          {
            a4 = *(_QWORD *)(v44 + 32);
            goto LABEL_24;
          }
        }
      }
LABEL_39:
      v8 = v45;
      sub_2E88D70(v45, v39[1] - 600LL, j, a4, i, a6);
      v22 = *(_DWORD **)(v45 + 32);
      if ( *(_WORD *)(v45 + 68) == 14 )
      {
        v23 = v22 + 10;
      }
      else
      {
        v23 = &v22[10 * (*(_DWORD *)(v45 + 40) & 0xFFFFFF)];
        v22 += 20;
      }
      for ( ; v23 != v22; v22 += 10 )
      {
        if ( !*(_BYTE *)v22 )
        {
          sub_2EAB0C0(v22, 0);
          *v22 &= 0xFFF000FF;
        }
      }
      goto LABEL_8;
    }
LABEL_10:
    v37 = *(_QWORD *)(v37 + 8);
  }
  while ( v35 != v37 );
  v9 = v48;
  v10 = 12LL * v50;
  return sub_C7D6A0(v9, v10, 4);
}
