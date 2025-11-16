// Function: sub_2579020
// Address: 0x2579020
//
char __fastcall sub_2579020(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r13
  bool v4; // cc
  unsigned __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // rbx
  __int64 v9; // r12
  __int64 v10; // rcx
  unsigned int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // rbx
  char v14; // r12
  __int64 v15; // r15
  __int64 v16; // rcx
  unsigned __int64 v17; // rcx
  __int64 *v18; // rdi
  _BYTE *v19; // r15
  unsigned int v20; // eax
  __int64 *v21; // r9
  __int64 v22; // r10
  int v23; // eax
  _BYTE *v24; // rdx
  __int64 *v25; // rax
  __int64 v26; // rax
  int v27; // r9d
  int v28; // r11d
  __int64 v29; // r10
  __int64 v30; // rdx
  _BYTE *v32; // [rsp+18h] [rbp-E8h]
  __int64 v33; // [rsp+20h] [rbp-E0h]
  __int64 v34; // [rsp+38h] [rbp-C8h]
  __int64 v35; // [rsp+48h] [rbp-B8h]
  __int64 v36; // [rsp+48h] [rbp-B8h]
  _BYTE **v37; // [rsp+58h] [rbp-A8h] BYREF
  void *v38; // [rsp+60h] [rbp-A0h] BYREF
  __int16 v39; // [rsp+68h] [rbp-98h]
  __int64 v40; // [rsp+70h] [rbp-90h] BYREF
  __int64 v41; // [rsp+78h] [rbp-88h]
  __int64 v42; // [rsp+80h] [rbp-80h]
  __int64 v43; // [rsp+88h] [rbp-78h]
  __int64 *v44; // [rsp+90h] [rbp-70h]
  __int64 v45; // [rsp+98h] [rbp-68h]
  _BYTE *v46; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v47; // [rsp+A8h] [rbp-58h]
  _BYTE v48[80]; // [rsp+B0h] [rbp-50h] BYREF

  v3 = (_QWORD *)(a1 + 72);
  v4 = (unsigned int)*(unsigned __int8 *)sub_250D070((_QWORD *)(a1 + 72)) - 12 <= 1;
  LOBYTE(v5) = *(_BYTE *)(a1 + 96);
  if ( v4 )
  {
    *(_BYTE *)(a1 + 97) = v5;
    return v5;
  }
  if ( *(_BYTE *)(a1 + 97) != (_BYTE)v5 )
  {
    v5 = sub_25096F0(v3);
    if ( v5 )
    {
      v6 = sub_25096F0(v3);
      LOBYTE(v5) = sub_B2FC80(v6);
      if ( !(_BYTE)v5 )
      {
        v5 = sub_2509740(v3);
        v7 = v5;
        if ( v5 )
        {
          v5 = *(_QWORD *)(*(_QWORD *)(a2 + 208) + 120LL);
          v34 = v5;
          if ( v5 )
          {
            v40 = 0;
            v41 = 0;
            v42 = 0;
            v43 = 0;
            v44 = (__int64 *)&v46;
            v45 = 0;
            v8 = *(_QWORD *)(sub_250D070(v3) + 16);
            if ( v8 )
            {
              v35 = v7;
              v9 = v8;
              do
              {
                v46 = (_BYTE *)v9;
                sub_25789E0((__int64)&v40, (__int64 *)&v46);
                v9 = *(_QWORD *)(v9 + 8);
              }
              while ( v9 );
              v7 = v35;
            }
            sub_2578B50(a1, a2, v34, v7, (__int64)&v40, a1 + 88);
            if ( *(_BYTE *)(a1 + 97) != *(_BYTE *)(a1 + 96) )
            {
              v46 = v48;
              v47 = 0x400000000LL;
              v37 = &v46;
              sub_2568920(v34, v7, (unsigned __int8 (__fastcall *)(__int64))sub_253B710, (__int64)&v37);
              v32 = &v46[8 * (unsigned int)v47];
              if ( v46 != v32 )
              {
                v33 = (__int64)v46;
                do
                {
                  v10 = *(_QWORD *)v33;
                  v11 = *(_DWORD *)(*(_QWORD *)v33 + 4LL) & 0x7FFFFFF;
                  v12 = 32LL * v11;
                  if ( (*(_BYTE *)(*(_QWORD *)v33 + 7LL) & 0x40) != 0 )
                  {
                    v13 = *(_QWORD *)(v10 - 8);
                    v36 = v13 + v12;
                    if ( v11 == 3 )
                      v13 += 32;
                  }
                  else
                  {
                    v36 = *(_QWORD *)v33;
                    v13 = v10 - v12;
                    v30 = v10 - v12 + 32;
                    if ( v11 == 3 )
                      v13 = v30;
                  }
                  if ( v13 == v36 )
                    goto LABEL_46;
                  v14 = 1;
                  do
                  {
                    v15 = (unsigned int)v45;
                    v16 = *(_QWORD *)(*(_QWORD *)v13 + 56LL);
                    v38 = &unk_4A16CD8;
                    v39 = 256;
                    if ( v16 )
                      v16 -= 24;
                    sub_2578B50(a1, a2, v34, v16, (__int64)&v40, (__int64)&v38);
                    v17 = (unsigned __int64)v44;
                    v18 = &v44[v15];
                    v19 = v18 + 1;
                    if ( v18 != &v44[(unsigned int)v45] )
                    {
                      do
                      {
                        if ( (_DWORD)v43 )
                        {
                          v20 = (v43 - 1) & (((unsigned int)*v18 >> 9) ^ ((unsigned int)*v18 >> 4));
                          v21 = (__int64 *)(v41 + 8LL * v20);
                          v22 = *v21;
                          if ( *v18 != *v21 )
                          {
                            v27 = 1;
                            if ( v22 == -4096 )
                              goto LABEL_27;
                            while ( 1 )
                            {
                              v28 = v27 + 1;
                              v20 = (v43 - 1) & (v27 + v20);
                              v21 = (__int64 *)(v41 + 8LL * v20);
                              v29 = *v21;
                              if ( *v18 == *v21 )
                                break;
                              v27 = v28;
                              if ( v29 == -4096 )
                                goto LABEL_27;
                            }
                          }
                          *v21 = -8192;
                          v17 = (unsigned __int64)v44;
                          LODWORD(v42) = v42 - 1;
                          ++HIDWORD(v42);
                        }
LABEL_27:
                        v23 = v45;
                        v24 = (_BYTE *)(v17 + 8LL * (unsigned int)v45);
                        if ( v19 != v24 )
                        {
                          v25 = (__int64 *)memmove(v18, v19, v24 - v19);
                          v17 = (unsigned __int64)v44;
                          v18 = v25;
                          v23 = v45;
                        }
                        v26 = (unsigned int)(v23 - 1);
                        LODWORD(v45) = v26;
                      }
                      while ( v18 != (__int64 *)(v17 + 8 * v26) );
                    }
                    v14 &= v39;
                    v13 += 32;
                  }
                  while ( v36 != v13 );
                  if ( v14 )
LABEL_46:
                    *(_WORD *)(a1 + 96) = 257;
                  v33 += 8;
                }
                while ( v32 != (_BYTE *)v33 );
                v32 = v46;
              }
              if ( v32 != v48 )
                _libc_free((unsigned __int64)v32);
            }
            if ( v44 != (__int64 *)&v46 )
              _libc_free((unsigned __int64)v44);
            LOBYTE(v5) = sub_C7D6A0(v41, 8LL * (unsigned int)v43, 8);
          }
        }
      }
    }
  }
  return v5;
}
