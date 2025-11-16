// Function: sub_25906D0
// Address: 0x25906d0
//
unsigned __int64 __fastcall sub_25906D0(__int64 a1, _QWORD *a2)
{
  _QWORD *v3; // r13
  unsigned __int8 *v4; // rax
  unsigned __int64 result; // rax
  __int64 v6; // r12
  __int64 i; // rbx
  __int64 v8; // rcx
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rbx
  char v12; // r12
  __int64 v13; // r15
  __int64 v14; // rcx
  unsigned __int64 v15; // rcx
  __int64 *v16; // rdi
  _BYTE *v17; // r15
  unsigned int v18; // eax
  __int64 *v19; // r9
  __int64 v20; // r10
  int v21; // eax
  _BYTE *v22; // rdx
  __int64 *v23; // rax
  __int64 v24; // rax
  int v25; // r9d
  int v26; // r11d
  __int64 v27; // r10
  __int64 v28; // rdx
  _BYTE *v29; // [rsp+18h] [rbp-E8h]
  __int64 v30; // [rsp+20h] [rbp-E0h]
  __int64 v31; // [rsp+30h] [rbp-D0h]
  __int64 v32; // [rsp+48h] [rbp-B8h]
  _BYTE **v33; // [rsp+58h] [rbp-A8h] BYREF
  void *v34; // [rsp+60h] [rbp-A0h] BYREF
  __int16 v35; // [rsp+68h] [rbp-98h]
  __int64 v36; // [rsp+70h] [rbp-90h] BYREF
  __int64 v37; // [rsp+78h] [rbp-88h]
  __int64 v38; // [rsp+80h] [rbp-80h]
  __int64 v39; // [rsp+88h] [rbp-78h]
  __int64 *v40; // [rsp+90h] [rbp-70h]
  __int64 v41; // [rsp+98h] [rbp-68h]
  _BYTE *v42; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v43; // [rsp+A8h] [rbp-58h]
  _BYTE v44[80]; // [rsp+B0h] [rbp-50h] BYREF

  v3 = (_QWORD *)(a1 + 72);
  v4 = (unsigned __int8 *)sub_250D070((_QWORD *)(a1 + 72));
  if ( *sub_BD3990(v4, (__int64)a2) == 20 )
  {
    result = *(unsigned __int8 *)(a1 + 96);
    *(_BYTE *)(a1 + 97) = result;
  }
  else
  {
    result = sub_2509740(v3);
    v6 = result;
    if ( result )
    {
      result = *(_QWORD *)(a2[26] + 120LL);
      v31 = result;
      if ( result )
      {
        v36 = 0;
        v37 = 0;
        v38 = 0;
        v39 = 0;
        v40 = (__int64 *)&v42;
        v41 = 0;
        for ( i = *(_QWORD *)(sub_250D070(v3) + 16); i; i = *(_QWORD *)(i + 8) )
        {
          v42 = (_BYTE *)i;
          sub_25789E0((__int64)&v36, (__int64 *)&v42);
        }
        sub_2590330(a1, a2, v31, v6, (__int64)&v36, a1 + 88);
        if ( *(_BYTE *)(a1 + 97) != *(_BYTE *)(a1 + 96) )
        {
          v42 = v44;
          v43 = 0x400000000LL;
          v33 = &v42;
          sub_2568920(v31, v6, (unsigned __int8 (__fastcall *)(__int64))sub_253B680, (__int64)&v33);
          v29 = &v42[8 * (unsigned int)v43];
          if ( v42 != v29 )
          {
            v30 = (__int64)v42;
            while ( 1 )
            {
              v8 = *(_QWORD *)v30;
              v9 = *(_DWORD *)(*(_QWORD *)v30 + 4LL) & 0x7FFFFFF;
              v10 = 32LL * v9;
              if ( (*(_BYTE *)(*(_QWORD *)v30 + 7LL) & 0x40) != 0 )
              {
                v11 = *(_QWORD *)(v8 - 8);
                v32 = v11 + v10;
                if ( v9 == 3 )
                  v11 += 32;
                if ( v11 != v32 )
                  goto LABEL_13;
              }
              else
              {
                v32 = *(_QWORD *)v30;
                v11 = v8 - v10;
                v28 = v8 - v10 + 32;
                if ( v9 == 3 )
                  v11 = v28;
                if ( v11 != v32 )
                {
LABEL_13:
                  v12 = 1;
                  do
                  {
                    v13 = (unsigned int)v41;
                    v14 = *(_QWORD *)(*(_QWORD *)v11 + 56LL);
                    v34 = &unk_4A16CD8;
                    v35 = 256;
                    if ( v14 )
                      v14 -= 24;
                    sub_2590330(a1, a2, v31, v14, (__int64)&v36, (__int64)&v34);
                    v15 = (unsigned __int64)v40;
                    v16 = &v40[v13];
                    v17 = v16 + 1;
                    if ( v16 != &v40[(unsigned int)v41] )
                    {
                      do
                      {
                        if ( (_DWORD)v39 )
                        {
                          v18 = (v39 - 1) & (((unsigned int)*v16 >> 9) ^ ((unsigned int)*v16 >> 4));
                          v19 = (__int64 *)(v37 + 8LL * v18);
                          v20 = *v19;
                          if ( *v16 != *v19 )
                          {
                            v25 = 1;
                            if ( v20 == -4096 )
                              goto LABEL_20;
                            while ( 1 )
                            {
                              v26 = v25 + 1;
                              v18 = (v39 - 1) & (v25 + v18);
                              v19 = (__int64 *)(v37 + 8LL * v18);
                              v27 = *v19;
                              if ( *v16 == *v19 )
                                break;
                              v25 = v26;
                              if ( v27 == -4096 )
                                goto LABEL_20;
                            }
                          }
                          *v19 = -8192;
                          v15 = (unsigned __int64)v40;
                          LODWORD(v38) = v38 - 1;
                          ++HIDWORD(v38);
                        }
LABEL_20:
                        v21 = v41;
                        v22 = (_BYTE *)(v15 + 8LL * (unsigned int)v41);
                        if ( v22 != v17 )
                        {
                          v23 = (__int64 *)memmove(v16, v17, v22 - v17);
                          v15 = (unsigned __int64)v40;
                          v16 = v23;
                          v21 = v41;
                        }
                        v24 = (unsigned int)(v21 - 1);
                        LODWORD(v41) = v24;
                      }
                      while ( v16 != (__int64 *)(v15 + 8 * v24) );
                    }
                    v12 &= v35;
                    v11 += 32;
                  }
                  while ( v32 != v11 );
                  if ( !v12 )
                    goto LABEL_25;
                }
              }
              *(_WORD *)(a1 + 96) = 257;
LABEL_25:
              v30 += 8;
              if ( v29 == (_BYTE *)v30 )
              {
                v29 = v42;
                break;
              }
            }
          }
          if ( v29 != v44 )
            _libc_free((unsigned __int64)v29);
        }
        if ( v40 != (__int64 *)&v42 )
          _libc_free((unsigned __int64)v40);
        return sub_C7D6A0(v37, 8LL * (unsigned int)v39, 8);
      }
    }
  }
  return result;
}
