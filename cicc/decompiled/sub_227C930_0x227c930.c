// Function: sub_227C930
// Address: 0x227c930
//
__int64 __fastcall sub_227C930(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v5; // rax
  __int64 v6; // r14
  bool v7; // zf
  __int64 *v8; // rax
  __int64 v9; // r14
  __int64 *v10; // r9
  int v11; // esi
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // r10
  __int64 v15; // rdx
  char v16; // al
  __int64 v17; // r15
  char v18; // cl
  __int64 v19; // rsi
  int v20; // ecx
  unsigned int v21; // esi
  int v22; // edx
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 result; // rax
  int v26; // edx
  __int64 *v27; // r12
  __int64 v28; // rsi
  __int64 *v29; // rbx
  unsigned __int64 v30; // r14
  __int64 v31; // rdi
  __int64 v32; // rsi
  __int64 *v33; // r15
  __int64 v34; // r12
  __int64 *v35; // rsi
  int v36; // edx
  unsigned int v37; // edi
  __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // rsi
  unsigned int v42; // edi
  __int64 *v43; // rax
  __int64 v44; // r10
  __int64 v45; // r13
  __int64 v46; // rax
  _QWORD *v47; // rbx
  _QWORD *v48; // rax
  _QWORD *v49; // r12
  _QWORD *v50; // rax
  __int64 v51; // rdi
  _QWORD *v52; // rdi
  __int64 v53; // rdx
  __int64 v54; // rsi
  __int64 v55; // rbx
  __int64 v56; // rdi
  int v57; // edx
  int v58; // r8d
  int v59; // esi
  _QWORD *v60; // rdx
  unsigned int v61; // eax
  int v62; // eax
  int v63; // eax
  __int64 v64; // rdx
  __int64 v65; // rcx
  int v66; // r8d
  unsigned int v67; // r8d
  int v68; // r9d
  int v69; // edi
  __int64 v70; // [rsp+10h] [rbp-150h]
  __int64 v71; // [rsp+18h] [rbp-148h]
  __int64 *v74; // [rsp+38h] [rbp-128h]
  __int64 v75; // [rsp+48h] [rbp-118h] BYREF
  _QWORD v76[2]; // [rsp+50h] [rbp-110h] BYREF
  __int64 *v77; // [rsp+60h] [rbp-100h] BYREF
  char v78[8]; // [rsp+68h] [rbp-F8h] BYREF
  _QWORD v79[6]; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v80; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v81; // [rsp+A8h] [rbp-B8h]
  __int64 *v82; // [rsp+B0h] [rbp-B0h] BYREF
  unsigned int v83; // [rsp+B8h] [rbp-A8h]
  char v84; // [rsp+130h] [rbp-30h] BYREF

  if ( *(_DWORD *)(a3 + 72) != *(_DWORD *)(a3 + 68)
    || (result = sub_B19060(a3, (__int64)&unk_4F82400, a3, a4), !(_BYTE)result)
    && (result = sub_B19060(a3, (__int64)&unk_4FDADC8, v64, v65), !(_BYTE)result) )
  {
    v5 = (__int64 *)&v82;
    v80 = 0;
    v81 = 1;
    do
    {
      *v5 = -4096;
      v5 += 2;
    }
    while ( v5 != (__int64 *)&v84 );
    v76[0] = &v80;
    v6 = a1 + 32;
    v76[1] = a1 + 64;
    v75 = a2;
    v7 = (unsigned __int8)sub_227BB30(a1 + 32, &v75, &v77) == 0;
    v8 = v77;
    if ( !v7 )
    {
      v74 = v77 + 1;
      v9 = v77[1];
      if ( (__int64 *)v9 == v77 + 1 )
        goto LABEL_22;
      goto LABEL_14;
    }
    v79[0] = v77;
    v20 = *(_DWORD *)(a1 + 48);
    ++*(_QWORD *)(a1 + 32);
    v21 = *(_DWORD *)(a1 + 56);
    v22 = v20 + 1;
    if ( 4 * (v20 + 1) >= 3 * v21 )
    {
      v21 *= 2;
    }
    else if ( v21 - *(_DWORD *)(a1 + 52) - v22 > v21 >> 3 )
    {
      goto LABEL_19;
    }
    sub_227C6A0(v6, v21);
    sub_227BB30(v6, &v75, v79);
    v22 = *(_DWORD *)(a1 + 48) + 1;
    v8 = (__int64 *)v79[0];
LABEL_19:
    *(_DWORD *)(a1 + 48) = v22;
    if ( *v8 != -4096 )
      --*(_DWORD *)(a1 + 52);
    v23 = v75;
    v8[3] = 0;
    v74 = v8 + 1;
    *v8 = v23;
    v8[2] = (__int64)(v8 + 1);
    v8[1] = (__int64)(v8 + 1);
    v9 = v8[1];
    if ( (__int64 *)*v74 == v74 )
      goto LABEL_22;
    while ( 1 )
    {
LABEL_14:
      v17 = *(_QWORD *)(v9 + 16);
      v18 = v81 & 1;
      if ( (v81 & 1) != 0 )
      {
        v10 = (__int64 *)&v82;
        v11 = 7;
      }
      else
      {
        v19 = v83;
        v10 = v82;
        if ( !v83 )
          goto LABEL_30;
        v11 = v83 - 1;
      }
      v12 = v11 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v13 = &v10[2 * v12];
      v14 = *v13;
      if ( v17 != *v13 )
        break;
LABEL_9:
      v15 = 16;
      if ( !v18 )
        v15 = 2LL * v83;
      if ( v13 == &v10[v15] )
      {
        v16 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, _QWORD *))(**(_QWORD **)(v9 + 24) + 16LL))(
                *(_QWORD *)(v9 + 24),
                a2,
                a3,
                v76);
        v77 = (__int64 *)v17;
        v78[0] = v16;
        sub_BBCF50((__int64)v79, (__int64)&v80, (__int64 *)&v77, v78);
      }
      v9 = *(_QWORD *)v9;
      if ( (__int64 *)v9 == v74 )
      {
        result = (unsigned int)v81 >> 1;
        v33 = (__int64 *)*v74;
        if ( !((unsigned int)v81 >> 1) )
          goto LABEL_67;
        if ( v74 == v33 )
          goto LABEL_22;
        result = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
        while ( 1 )
        {
          v34 = v33[2];
          if ( (v81 & 1) != 0 )
          {
            v35 = (__int64 *)&v82;
            v36 = 7;
          }
          else
          {
            v35 = v82;
            if ( !v83 )
              goto LABEL_64;
            v36 = v83 - 1;
          }
          LODWORD(v71) = ((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4);
          v37 = v36 & v71;
          result = (__int64)&v35[2 * (v36 & (unsigned int)v71)];
          v38 = *(_QWORD *)result;
          if ( v34 == *(_QWORD *)result )
          {
LABEL_38:
            if ( *(_BYTE *)(result + 8) )
            {
              v39 = sub_227B160(a1, (__int64)&unk_4F8A320, a2);
              if ( v39 )
              {
                v40 = *(unsigned int *)(a1 + 24);
                v41 = *(_QWORD *)(a1 + 8);
                if ( (_DWORD)v40 )
                {
                  v42 = (v40 - 1) & v71;
                  v43 = (__int64 *)(v41 + 16LL * v42);
                  v44 = *v43;
                  if ( v34 == *v43 )
                    goto LABEL_42;
                  v63 = 1;
                  while ( v44 != -4096 )
                  {
                    v68 = v63 + 1;
                    v42 = (v40 - 1) & (v63 + v42);
                    v43 = (__int64 *)(v41 + 16LL * v42);
                    v44 = *v43;
                    if ( v34 == *v43 )
                      goto LABEL_42;
                    v63 = v68;
                  }
                }
                v43 = (__int64 *)(v41 + 16 * v40);
LABEL_42:
                v45 = v43[1];
                v46 = *(_QWORD *)(v39 + 8);
                if ( v46 )
                {
                  v47 = *(_QWORD **)(v46 + 1008);
                  v48 = &v47[4 * *(unsigned int *)(v46 + 1016)];
                  if ( v47 != v48 )
                  {
                    v70 = v34;
                    v49 = v48;
                    do
                    {
                      v79[0] = 0;
                      v50 = (_QWORD *)sub_22077B0(0x10u);
                      if ( v50 )
                      {
                        v50[1] = a2;
                        *v50 = &unk_4A08BA8;
                      }
                      v51 = v79[0];
                      v79[0] = v50;
                      if ( v51 )
                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v51 + 8LL))(v51);
                      v52 = v47;
                      v54 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v45 + 24LL))(v45);
                      if ( (v47[3] & 2) == 0 )
                        v52 = (_QWORD *)*v47;
                      (*(void (__fastcall **)(_QWORD *, __int64, __int64, _QWORD *))(v47[3] & 0xFFFFFFFFFFFFFFF8LL))(
                        v52,
                        v54,
                        v53,
                        v79);
                      if ( v79[0] )
                        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v79[0] + 8LL))(v79[0]);
                      v47 += 4;
                    }
                    while ( v49 != v47 );
                    v34 = v70;
                  }
                }
              }
              v55 = *v33;
              --v74[2];
              sub_2208CA0(v33);
              v56 = v33[3];
              if ( v56 )
                (*(void (__fastcall **)(__int64))(*(_QWORD *)v56 + 8LL))(v56);
              j_j___libc_free_0((unsigned __int64)v33);
              result = a1;
              v57 = *(_DWORD *)(a1 + 88);
              if ( v57 )
              {
                v58 = 1;
                v59 = v57 - 1;
                for ( result = (v57 - 1)
                             & ((unsigned int)((0xBF58476D1CE4E5B9LL
                                              * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                                               | (unsigned __int64)(v71 << 32))) >> 31)
                              ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; result = v59 & v61 )
                {
                  v60 = (_QWORD *)(*(_QWORD *)(a1 + 72) + 24LL * (unsigned int)result);
                  if ( v34 == *v60 && a2 == v60[1] )
                    break;
                  if ( *v60 == -4096 && v60[1] == -4096 )
                    goto LABEL_75;
                  v61 = v58 + result;
                  ++v58;
                }
                result = a1;
                *v60 = -8192;
                v60[1] = -8192;
                --*(_DWORD *)(a1 + 80);
                ++*(_DWORD *)(a1 + 84);
              }
LABEL_75:
              v33 = (__int64 *)v55;
              goto LABEL_65;
            }
          }
          else
          {
            result = 1;
            while ( v38 != -4096 )
            {
              v67 = result + 1;
              v37 = v36 & (result + v37);
              result = (__int64)&v35[2 * v37];
              v38 = *(_QWORD *)result;
              if ( v34 == *(_QWORD *)result )
                goto LABEL_38;
              result = v67;
            }
          }
LABEL_64:
          v33 = (__int64 *)*v33;
LABEL_65:
          if ( v33 == v74 )
          {
            v33 = (__int64 *)*v33;
LABEL_67:
            if ( v74 != v33 )
            {
LABEL_68:
              if ( (v81 & 1) == 0 )
                return sub_C7D6A0((__int64)v82, 16LL * v83, 8);
              return result;
            }
LABEL_22:
            v24 = *(_QWORD *)(a1 + 40);
            result = *(unsigned int *)(a1 + 56);
            if ( (_DWORD)result )
            {
              v26 = result - 1;
              result = ((_DWORD)result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
              v27 = (__int64 *)(v24 + 32 * result);
              v28 = *v27;
              if ( a2 == *v27 )
              {
LABEL_24:
                v29 = (__int64 *)v27[1];
                while ( v27 + 1 != v29 )
                {
                  v30 = (unsigned __int64)v29;
                  v29 = (__int64 *)*v29;
                  v31 = *(_QWORD *)(v30 + 24);
                  if ( v31 )
                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v31 + 8LL))(v31);
                  j_j___libc_free_0(v30);
                }
                result = a1;
                *v27 = -8192;
                --*(_DWORD *)(a1 + 48);
                ++*(_DWORD *)(a1 + 52);
              }
              else
              {
                v69 = 1;
                while ( v28 != -4096 )
                {
                  result = v26 & (unsigned int)(v69 + result);
                  v27 = (__int64 *)(v24 + 32LL * (unsigned int)result);
                  v28 = *v27;
                  if ( a2 == *v27 )
                    goto LABEL_24;
                  ++v69;
                }
              }
            }
            goto LABEL_68;
          }
        }
      }
    }
    v62 = 1;
    while ( v14 != -4096 )
    {
      v66 = v62 + 1;
      v12 = v11 & (v62 + v12);
      v13 = &v10[2 * v12];
      v14 = *v13;
      if ( v17 == *v13 )
        goto LABEL_9;
      v62 = v66;
    }
    if ( v18 )
    {
      v32 = 16;
    }
    else
    {
      v19 = v83;
LABEL_30:
      v32 = 2 * v19;
    }
    v13 = &v10[v32];
    goto LABEL_9;
  }
  return result;
}
