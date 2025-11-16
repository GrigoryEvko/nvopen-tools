// Function: sub_1AA8CA0
// Address: 0x1aa8ca0
//
__int64 __fastcall sub_1AA8CA0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v5; // r12
  int v7; // ecx
  unsigned int v8; // ecx
  __int64 v9; // rdx
  __int64 v10; // r12
  int v11; // eax
  int v12; // esi
  __int64 v13; // rcx
  unsigned int v14; // edx
  _QWORD *v15; // rax
  _QWORD *v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rcx
  __int64 v19; // r9
  int v20; // esi
  unsigned int v21; // r8d
  unsigned int v22; // edi
  _QWORD *v23; // rax
  _QWORD *v24; // rdx
  __int64 v25; // rax
  const void *v26; // r10
  __int64 v27; // rax
  unsigned int v28; // r8d
  _QWORD *v29; // rax
  _QWORD *v30; // rdx
  __int64 v31; // rbx
  __int64 v32; // r15
  _BYTE *v33; // rsi
  __int64 *v34; // rax
  __int64 v35; // r14
  __int64 *v36; // rbx
  __int64 v37; // rdi
  __int64 v38; // r14
  __int64 v39; // rdi
  __int64 v40; // r12
  __int64 *v41; // r15
  _BYTE *v42; // rsi
  __int64 v43; // r14
  __int64 v44; // rax
  _BYTE *v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rcx
  int v48; // r8d
  int v49; // r9d
  int v51; // eax
  int v52; // r10d
  __int64 v53; // rax
  int v54; // eax
  int v55; // r8d
  int v56; // eax
  int v57; // edi
  unsigned int v58; // [rsp+Ch] [rbp-A4h]
  void *v59; // [rsp+10h] [rbp-A0h]
  __int64 v60; // [rsp+10h] [rbp-A0h]
  __int64 v61; // [rsp+10h] [rbp-A0h]
  void *srca; // [rsp+18h] [rbp-98h]
  unsigned int srcb; // [rsp+18h] [rbp-98h]
  void *src; // [rsp+18h] [rbp-98h]
  size_t n; // [rsp+20h] [rbp-90h]
  __int64 *dest; // [rsp+28h] [rbp-88h]
  __int64 *v67; // [rsp+30h] [rbp-80h]
  __int64 v69; // [rsp+48h] [rbp-68h] BYREF
  _QWORD v70[2]; // [rsp+50h] [rbp-60h] BYREF
  __int64 v71[2]; // [rsp+60h] [rbp-50h] BYREF
  __int16 v72; // [rsp+70h] [rbp-40h]

  v5 = (__int64 *)(a2 + 24);
  while ( 1 )
  {
    v7 = *((unsigned __int8 *)v5 - 8);
    if ( (_BYTE)v7 != 77 )
    {
      v8 = v7 - 34;
      if ( v8 > 0x36 || ((1LL << v8) & 0x40018000000001LL) == 0 )
        break;
    }
    v5 = (__int64 *)v5[1];
    if ( !v5 )
      BUG();
  }
  v71[0] = (__int64)v70;
  v70[0] = sub_1649960((__int64)a1);
  v70[1] = v9;
  v72 = 773;
  v71[1] = (__int64)".split";
  v10 = sub_157FBF0(a1, v5, (__int64)v71);
  if ( a4 )
  {
    v11 = *(_DWORD *)(a4 + 24);
    if ( v11 )
    {
      v12 = v11 - 1;
      v13 = *(_QWORD *)(a4 + 8);
      v14 = (v11 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v15 = (_QWORD *)(v13 + 16LL * v14);
      v16 = (_QWORD *)*v15;
      if ( a1 == (_QWORD *)*v15 )
      {
LABEL_8:
        v17 = v15[1];
        if ( v17 )
          sub_1400330(v17, v10, a4);
      }
      else
      {
        v54 = 1;
        while ( v16 != (_QWORD *)-8LL )
        {
          v55 = v54 + 1;
          v14 = v12 & (v54 + v14);
          v15 = (_QWORD *)(v13 + 16LL * v14);
          v16 = (_QWORD *)*v15;
          if ( a1 == (_QWORD *)*v15 )
            goto LABEL_8;
          v54 = v55;
        }
      }
    }
  }
  if ( a3 )
  {
    v18 = *(unsigned int *)(a3 + 48);
    if ( (_DWORD)v18 )
    {
      v19 = *(_QWORD *)(a3 + 32);
      v20 = v18 - 1;
      v21 = ((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4);
      v22 = (v18 - 1) & v21;
      v23 = (_QWORD *)(v19 + 16LL * v22);
      v24 = (_QWORD *)*v23;
      if ( a1 == (_QWORD *)*v23 )
      {
LABEL_13:
        if ( v23 == (_QWORD *)(v19 + 16LL * (unsigned int)v18) )
          return v10;
        v25 = v23[1];
        if ( !v25 )
          return v10;
        v26 = *(const void **)(v25 + 24);
        n = *(_QWORD *)(v25 + 32) - (_QWORD)v26;
        if ( n > 0x7FFFFFFFFFFFFFF8LL )
          sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
        if ( n )
        {
          v59 = *(void **)(v25 + 32);
          srca = *(void **)(v25 + 24);
          v27 = sub_22077B0(n);
          v26 = srca;
          dest = (__int64 *)v27;
          v21 = ((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4);
          v19 = *(_QWORD *)(a3 + 32);
          v18 = *(unsigned int *)(a3 + 48);
          v67 = (__int64 *)(n + v27);
          if ( srca == v59 )
          {
LABEL_19:
            v69 = v10;
            if ( !(_DWORD)v18 )
              goto LABEL_69;
            v20 = v18 - 1;
LABEL_21:
            v28 = v20 & v21;
            v29 = (_QWORD *)(v19 + 16LL * v28);
            v30 = (_QWORD *)*v29;
            if ( a1 == (_QWORD *)*v29 )
            {
LABEL_22:
              if ( v29 != (_QWORD *)(16 * v18 + v19) )
              {
                v31 = v29[1];
                *(_BYTE *)(a3 + 72) = 0;
                sub_1AA56D0(v71, v10, v31);
                v32 = v71[0];
                v70[0] = v71[0];
                v33 = *(_BYTE **)(v31 + 32);
                if ( v33 == *(_BYTE **)(v31 + 40) )
                {
                  sub_15CE310(v31 + 24, v33, v70);
                  v32 = v71[0];
                }
                else
                {
                  if ( v33 )
                  {
                    *(_QWORD *)v33 = v71[0];
                    v33 = *(_BYTE **)(v31 + 32);
                    v32 = v71[0];
                  }
                  *(_QWORD *)(v31 + 32) = v33 + 8;
                }
                v71[0] = 0;
                v34 = sub_15CFF10(a3 + 24, &v69);
                v35 = v34[1];
                v36 = v34;
                v34[1] = v32;
                if ( v35 )
                {
                  v37 = *(_QWORD *)(v35 + 24);
                  if ( v37 )
                    j_j___libc_free_0(v37, *(_QWORD *)(v35 + 40) - v37);
                  j_j___libc_free_0(v35, 56);
                  v32 = v36[1];
                }
                v38 = v71[0];
                if ( v71[0] )
                {
                  v39 = *(_QWORD *)(v71[0] + 24);
                  if ( v39 )
                    j_j___libc_free_0(v39, *(_QWORD *)(v71[0] + 40) - v39);
                  j_j___libc_free_0(v38, 56);
                }
                src = (void *)(v32 + 24);
                if ( v67 != dest )
                {
                  v61 = v10;
                  v40 = v32;
                  v41 = dest;
                  do
                  {
                    v43 = *v41;
                    *(_BYTE *)(a3 + 72) = 0;
                    v44 = *(_QWORD *)(v43 + 8);
                    if ( v44 != v40 )
                    {
                      v71[0] = v43;
                      v45 = sub_1AA5610(*(_QWORD **)(v44 + 24), *(_QWORD *)(v44 + 32), v71);
                      sub_15CDF70(*(_QWORD *)(v43 + 8) + 24LL, v45);
                      *(_QWORD *)(v43 + 8) = v40;
                      v71[0] = v43;
                      v42 = *(_BYTE **)(v40 + 32);
                      if ( v42 == *(_BYTE **)(v40 + 40) )
                      {
                        sub_15CE310((__int64)src, v42, v71);
                      }
                      else
                      {
                        if ( v42 )
                        {
                          *(_QWORD *)v42 = v43;
                          v42 = *(_BYTE **)(v40 + 32);
                        }
                        v42 += 8;
                        *(_QWORD *)(v40 + 32) = v42;
                      }
                      if ( *(_DWORD *)(v43 + 16) != *(_DWORD *)(*(_QWORD *)(v43 + 8) + 16LL) + 1 )
                        sub_1AA5500(v43, (__int64)v42, v46, v47, v48, v49);
                    }
                    ++v41;
                  }
                  while ( v67 != v41 );
                  v10 = v61;
                }
                if ( dest )
                  j_j___libc_free_0(dest, n);
                return v10;
              }
            }
            else
            {
              v56 = 1;
              while ( v30 != (_QWORD *)-8LL )
              {
                v57 = v56 + 1;
                v28 = v20 & (v56 + v28);
                v29 = (_QWORD *)(v19 + 16LL * v28);
                v30 = (_QWORD *)*v29;
                if ( a1 == (_QWORD *)*v29 )
                  goto LABEL_22;
                v56 = v57;
              }
            }
LABEL_69:
            *(_BYTE *)(a3 + 72) = 0;
            sub_1AA56D0(v71, v10, 0);
            v70[0] = v71[0];
            BUG();
          }
        }
        else
        {
          if ( v26 == *(const void **)(v25 + 32) )
          {
            v69 = v10;
            v67 = 0;
            dest = 0;
            goto LABEL_21;
          }
          v67 = 0;
          dest = 0;
        }
        v58 = v18;
        v60 = v19;
        srcb = v21;
        memcpy(dest, v26, n);
        v18 = v58;
        v19 = v60;
        v21 = srcb;
        goto LABEL_19;
      }
      v51 = 1;
      while ( v24 != (_QWORD *)-8LL )
      {
        v52 = v51 + 1;
        v53 = v20 & (v22 + v51);
        v22 = v53;
        v23 = (_QWORD *)(v19 + 16 * v53);
        v24 = (_QWORD *)*v23;
        if ( a1 == (_QWORD *)*v23 )
          goto LABEL_13;
        v51 = v52;
      }
    }
  }
  return v10;
}
