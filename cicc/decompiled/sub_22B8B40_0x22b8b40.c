// Function: sub_22B8B40
// Address: 0x22b8b40
//
__int64 __fastcall sub_22B8B40(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 *v9; // rdx
  __int64 *v10; // r12
  __int64 *v11; // rbx
  __int64 v12; // rcx
  int v13; // r9d
  __int64 v14; // rdi
  unsigned int v15; // esi
  __int64 *v16; // rdx
  __int64 v17; // r10
  __int64 v18; // rax
  int v19; // edx
  __int64 v20; // rdi
  unsigned int v21; // esi
  int *v22; // rcx
  int v23; // r10d
  __int64 v24; // rax
  int v25; // edx
  __int64 v26; // rdi
  unsigned int v27; // esi
  int *v28; // rcx
  int v29; // r10d
  __int64 v30; // rdx
  int v31; // ecx
  __int64 v32; // rsi
  unsigned int v33; // edi
  int *v34; // rax
  int v35; // r10d
  __int64 v36; // rdx
  __int64 v37; // rdi
  __int64 v38; // rsi
  unsigned int v39; // ecx
  __int64 *v40; // rax
  __int64 v41; // r10
  int v42; // edx
  __int64 v43; // rax
  __int64 v44; // rdi
  unsigned int v45; // esi
  int *v46; // rcx
  int v47; // r10d
  int v48; // eax
  int v49; // eax
  int v50; // eax
  int v51; // ecx
  int v52; // edx
  int v53; // ecx
  int v54; // ecx
  __int64 v55; // [rsp-A0h] [rbp-A0h]
  int v56; // [rsp-98h] [rbp-98h]
  int v57; // [rsp-94h] [rbp-94h]
  int v58; // [rsp-80h] [rbp-80h]
  int v59; // [rsp-80h] [rbp-80h]
  int v60; // [rsp-80h] [rbp-80h]
  int v61; // [rsp-80h] [rbp-80h]
  int v62; // [rsp-80h] [rbp-80h]
  int v63; // [rsp-80h] [rbp-80h]
  int v64; // [rsp-70h] [rbp-70h] BYREF
  int v65; // [rsp-6Ch] [rbp-6Ch] BYREF
  _BYTE v66[104]; // [rsp-68h] [rbp-68h] BYREF

  result = *(unsigned int *)(a1 + 40);
  if ( (_DWORD)result )
  {
    v9 = *(__int64 **)(a1 + 32);
    v10 = &v9[2 * *(unsigned int *)(a1 + 48)];
    if ( v9 != v10 )
    {
      while ( 1 )
      {
        result = *v9;
        v11 = v9;
        if ( *v9 != -4096 && result != -8192 )
          break;
        v9 += 2;
        if ( v10 == v9 )
          return result;
      }
      while ( v10 != v11 )
      {
        v12 = *(unsigned int *)(a4 + 48);
        v13 = *((_DWORD *)v11 + 2);
        v14 = *(_QWORD *)(a4 + 32);
        if ( !(_DWORD)v12 )
          goto LABEL_62;
        v15 = (v12 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
        v16 = (__int64 *)(v14 + 16LL * v15);
        v17 = *v16;
        if ( *v16 != result )
        {
          v52 = 1;
          while ( v17 != -4096 )
          {
            v15 = (v12 - 1) & (v52 + v15);
            v61 = v52 + 1;
            v16 = (__int64 *)(v14 + 16LL * v15);
            v17 = *v16;
            if ( *v16 == result )
              goto LABEL_10;
            v52 = v61;
          }
LABEL_62:
          abort();
        }
LABEL_10:
        if ( v16 == (__int64 *)(v14 + 16 * v12) )
          goto LABEL_62;
        v18 = *(unsigned int *)(a4 + 112);
        v19 = *((_DWORD *)v16 + 2);
        v20 = *(_QWORD *)(a4 + 96);
        if ( !(_DWORD)v18 )
          goto LABEL_63;
        v21 = (v18 - 1) & (37 * v19);
        v22 = (int *)(v20 + 8LL * v21);
        v23 = *v22;
        if ( v19 != *v22 )
        {
          v54 = 1;
          while ( v23 != -1 )
          {
            v21 = (v18 - 1) & (v54 + v21);
            v63 = v54 + 1;
            v22 = (int *)(v20 + 8LL * v21);
            v23 = *v22;
            if ( v19 == *v22 )
              goto LABEL_13;
            v54 = v63;
          }
LABEL_63:
          abort();
        }
LABEL_13:
        if ( v22 == (int *)(v20 + 8 * v18) )
          goto LABEL_63;
        v24 = *(unsigned int *)(a3 + 144);
        v25 = v22[1];
        v26 = *(_QWORD *)(a3 + 128);
        if ( !(_DWORD)v24 )
          goto LABEL_66;
        v27 = (v24 - 1) & (37 * v25);
        v28 = (int *)(v26 + 8LL * v27);
        v29 = *v28;
        if ( v25 != *v28 )
        {
          v53 = 1;
          while ( v29 != -1 )
          {
            v27 = (v24 - 1) & (v53 + v27);
            v62 = v53 + 1;
            v28 = (int *)(v26 + 8LL * v27);
            v29 = *v28;
            if ( v25 == *v28 )
              goto LABEL_16;
            v53 = v62;
          }
LABEL_66:
          abort();
        }
LABEL_16:
        if ( v28 == (int *)(v26 + 8 * v24) )
          goto LABEL_66;
        v30 = *(unsigned int *)(a3 + 80);
        v31 = v28[1];
        v32 = *(_QWORD *)(a3 + 64);
        if ( !(_DWORD)v30 )
          goto LABEL_67;
        v33 = (v30 - 1) & (37 * v31);
        v34 = (int *)(v32 + 16LL * v33);
        v35 = *v34;
        if ( *v34 != v31 )
        {
          v50 = 1;
          while ( v35 != -1 )
          {
            v33 = (v30 - 1) & (v50 + v33);
            v59 = v50 + 1;
            v34 = (int *)(v32 + 16LL * v33);
            v35 = *v34;
            if ( v31 == *v34 )
              goto LABEL_19;
            v50 = v59;
          }
LABEL_67:
          abort();
        }
LABEL_19:
        if ( v34 == (int *)(v32 + 16 * v30) )
          goto LABEL_67;
        v36 = *(unsigned int *)(a2 + 48);
        v37 = *((_QWORD *)v34 + 1);
        v38 = *(_QWORD *)(a2 + 32);
        if ( !(_DWORD)v36 )
          goto LABEL_64;
        v39 = (v36 - 1) & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
        v40 = (__int64 *)(v38 + 16LL * v39);
        v41 = *v40;
        if ( *v40 != v37 )
        {
          v49 = 1;
          while ( v41 != -4096 )
          {
            v39 = (v36 - 1) & (v49 + v39);
            v58 = v49 + 1;
            v40 = (__int64 *)(v38 + 16LL * v39);
            v41 = *v40;
            if ( v37 == *v40 )
              goto LABEL_22;
            v49 = v58;
          }
LABEL_64:
          abort();
        }
LABEL_22:
        if ( v40 == (__int64 *)(v38 + 16 * v36) )
          goto LABEL_64;
        v42 = *((_DWORD *)v40 + 2);
        v43 = *(unsigned int *)(a2 + 112);
        v44 = *(_QWORD *)(a2 + 96);
        if ( !(_DWORD)v43 )
          goto LABEL_65;
        v45 = (v43 - 1) & (37 * v42);
        v46 = (int *)(v44 + 8LL * v45);
        v47 = *v46;
        if ( v42 != *v46 )
        {
          v51 = 1;
          while ( v47 != -1 )
          {
            v45 = (v43 - 1) & (v51 + v45);
            v60 = v51 + 1;
            v46 = (int *)(v44 + 8LL * v45);
            v47 = *v46;
            if ( v42 == *v46 )
              goto LABEL_25;
            v51 = v60;
          }
LABEL_65:
          abort();
        }
LABEL_25:
        if ( v46 == (int *)(v44 + 8 * v43) )
          goto LABEL_65;
        v48 = v46[1];
        v65 = *((_DWORD *)v11 + 2);
        v11 += 2;
        v55 = a4;
        v64 = v48;
        v56 = v48;
        v57 = v13;
        sub_22B89D0((__int64)v66, a1 + 120, &v64, &v65);
        v64 = v57;
        v65 = v56;
        result = sub_22B89D0((__int64)v66, a1 + 88, &v64, &v65);
        if ( v11 == v10 )
          return result;
        a4 = v55;
        while ( 1 )
        {
          result = *v11;
          if ( *v11 != -8192 && result != -4096 )
            break;
          v11 += 2;
          if ( v10 == v11 )
            return result;
        }
      }
    }
  }
  return result;
}
