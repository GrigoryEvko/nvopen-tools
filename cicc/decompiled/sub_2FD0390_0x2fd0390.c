// Function: sub_2FD0390
// Address: 0x2fd0390
//
__int64 *__fastcall sub_2FD0390(__int64 a1, int *a2, __int64 a3)
{
  int v4; // r12d
  int *v5; // rbx
  __int64 v6; // r13
  __int64 *v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r8
  _QWORD *v10; // r9
  __int64 *v11; // rax
  __int64 *v12; // r15
  __int64 *v13; // r15
  __int64 *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r8
  _QWORD *v17; // rbx
  _QWORD *v18; // rdi
  unsigned __int64 v19; // rdi
  _QWORD *v20; // rax
  unsigned __int64 v21; // rcx
  __int64 *result; // rax
  __int64 v23; // rdx
  unsigned __int64 v24; // rdx
  _QWORD *v25; // rax
  _QWORD *v26; // r8
  int v27; // r9d
  unsigned __int64 v28; // rsi
  int v29; // r15d
  unsigned __int64 v30; // rdx
  unsigned __int64 v31; // rax
  __int64 **v32; // r10
  __int64 v33; // r13
  __int64 *v34; // rax
  int v35; // ecx
  char v36; // al
  unsigned __int64 v37; // rdx
  __int64 *v38; // r8
  unsigned __int64 v39; // r9
  _QWORD *v40; // r10
  __int64 ***v41; // rax
  __int64 *v42; // rdx
  size_t v43; // r15
  void *v44; // rax
  _QWORD *v45; // rax
  __int64 *v46; // rsi
  unsigned __int64 v47; // rdi
  __int64 *v48; // rcx
  unsigned __int64 v49; // rdx
  __int64 ***v50; // rax
  __int64 v51; // rax
  __int64 *v52; // rdx
  unsigned __int64 v53; // [rsp+8h] [rbp-B8h]
  __int64 *v54; // [rsp+10h] [rbp-B0h]
  __int64 *v55; // [rsp+10h] [rbp-B0h]
  __int64 *v56; // [rsp+18h] [rbp-A8h]
  unsigned __int64 v57; // [rsp+18h] [rbp-A8h]
  _QWORD *v58; // [rsp+18h] [rbp-A8h]
  int v59; // [rsp+30h] [rbp-90h]
  unsigned __int64 v60; // [rsp+30h] [rbp-90h]
  int *v61; // [rsp+38h] [rbp-88h]
  int v62; // [rsp+4Ch] [rbp-74h] BYREF
  _QWORD *v63; // [rsp+50h] [rbp-70h] BYREF
  unsigned __int64 v64; // [rsp+58h] [rbp-68h]
  __int64 *v65; // [rsp+60h] [rbp-60h] BYREF
  __int64 v66; // [rsp+68h] [rbp-58h]
  __m128i v67; // [rsp+70h] [rbp-50h] BYREF
  _QWORD v68[8]; // [rsp+80h] [rbp-40h] BYREF

  v4 = 0;
  v5 = a2;
  v63 = v68;
  v64 = 1;
  v65 = 0;
  v66 = 0;
  v67.m128i_i32[0] = 1065353216;
  v67.m128i_i64[1] = 0;
  v68[0] = 0;
  v61 = &a2[a3];
  if ( a2 != v61 )
  {
    do
    {
      if ( *v5 != -1 )
      {
        v11 = sub_2FD0320(&v63, v5);
        v12 = v11;
        if ( !v11 )
        {
          v59 = *v5;
          v13 = sub_2FD0320(&v63, v5);
          if ( v13 )
          {
LABEL_8:
            v62 = v4;
            v14 = sub_2FD0320((_QWORD *)(a1 + 104), &v62);
            sub_2F68500((__int64)(v13 + 2), v14 + 2, (__int64 *)(a1 + 8), v15, v16);
            goto LABEL_4;
          }
          v25 = (_QWORD *)sub_22077B0(0x88u);
          v26 = v25;
          if ( v25 )
            *v25 = 0;
          v27 = *v5;
          v28 = v64;
          v25[2] = v25 + 4;
          v29 = v59;
          v25[10] = v25 + 12;
          v60 = v27;
          v30 = v27 % v28;
          v31 = (unsigned __int64)v63;
          *((_DWORD *)v26 + 2) = v27;
          v26[3] = 0x200000000LL;
          v26[11] = 0x200000000LL;
          v26[14] = 0;
          v26[15] = 0;
          v26[16] = (unsigned int)(v29 + 0x40000000);
          v32 = *(__int64 ***)(v31 + 8 * v30);
          v33 = v30;
          if ( v32 )
          {
            v34 = *v32;
            if ( v27 == *((_DWORD *)*v32 + 2) )
            {
LABEL_25:
              v13 = *v32;
              if ( *v32 )
              {
                sub_2FCF520(v26);
                goto LABEL_8;
              }
            }
            else
            {
              while ( *v34 )
              {
                v35 = *(_DWORD *)(*v34 + 8);
                v32 = (__int64 **)v34;
                if ( v27 % v28 != v35 % v28 )
                  break;
                v34 = (__int64 *)*v34;
                if ( v27 == v35 )
                  goto LABEL_25;
              }
            }
          }
          v56 = v26;
          v36 = sub_222DA10((__int64)&v67, v28, v66, 1);
          v38 = v56;
          v39 = v37;
          if ( v36 )
          {
            if ( v37 == 1 )
            {
              v68[0] = 0;
              v40 = v68;
            }
            else
            {
              if ( v37 > 0xFFFFFFFFFFFFFFFLL )
                sub_4261EA(&v67, v28, v37);
              v43 = 8 * v37;
              v54 = v56;
              v57 = v37;
              v44 = (void *)sub_22077B0(8 * v37);
              v45 = memset(v44, 0, v43);
              v38 = v54;
              v39 = v57;
              v40 = v45;
            }
            v46 = v65;
            v65 = 0;
            if ( v46 )
            {
              v47 = 0;
              do
              {
                v48 = v46;
                v46 = (__int64 *)*v46;
                v49 = *((int *)v48 + 2) % v39;
                v50 = (__int64 ***)&v40[v49];
                if ( *v50 )
                {
                  *v48 = (__int64)**v50;
                  **v50 = v48;
                }
                else
                {
                  *v48 = (__int64)v65;
                  v65 = v48;
                  *v50 = &v65;
                  if ( *v48 )
                    v40[v47] = v48;
                  v47 = v49;
                }
              }
              while ( v46 );
            }
            if ( v63 != v68 )
            {
              v53 = v39;
              v55 = v38;
              v58 = v40;
              j_j___libc_free_0((unsigned __int64)v63);
              v39 = v53;
              v38 = v55;
              v40 = v58;
            }
            v64 = v39;
            v63 = v40;
            v33 = v60 % v39;
          }
          else
          {
            v40 = v63;
          }
          v41 = (__int64 ***)&v40[v33];
          v42 = (__int64 *)v40[v33];
          if ( v42 )
          {
            *v38 = *v42;
            **v41 = v38;
          }
          else
          {
            v52 = v65;
            v65 = v38;
            *v38 = (__int64)v52;
            if ( v52 )
            {
              v40[*((int *)v52 + 2) % v64] = v38;
              v41 = (__int64 ***)&v63[v33];
            }
            *v41 = &v65;
          }
          ++v66;
          v13 = v38;
          goto LABEL_8;
        }
        v6 = *(_QWORD *)v11[10];
        v62 = v4;
        v7 = sub_2FD0320((_QWORD *)(a1 + 104), &v62);
        sub_2E0FDD0((__int64)(v12 + 2), (__int64)(v7 + 2), v6, v8, v9, v10);
      }
LABEL_4:
      ++v4;
      ++v5;
    }
    while ( v61 != v5 );
  }
  v17 = *(_QWORD **)(a1 + 120);
  while ( v17 )
  {
    v18 = v17;
    v17 = (_QWORD *)*v17;
    sub_2FCF520(v18);
  }
  v19 = *(_QWORD *)(a1 + 104);
  if ( v19 != a1 + 152 )
    j_j___libc_free_0(v19);
  v20 = v63;
  *(__m128i *)(a1 + 136) = _mm_loadu_si128(&v67);
  if ( v20 == v68 )
  {
    v51 = v68[0];
    *(_QWORD *)(a1 + 104) = a1 + 152;
    *(_QWORD *)(a1 + 152) = v51;
  }
  else
  {
    *(_QWORD *)(a1 + 104) = v20;
  }
  v21 = v64;
  result = v65;
  v23 = v66;
  *(_QWORD *)(a1 + 112) = v64;
  *(_QWORD *)(a1 + 120) = result;
  *(_QWORD *)(a1 + 128) = v23;
  if ( result )
  {
    v24 = *((int *)result + 2) % v21;
    result = *(__int64 **)(a1 + 104);
    result[v24] = a1 + 120;
  }
  return result;
}
