// Function: sub_2F50AD0
// Address: 0x2f50ad0
//
void __fastcall sub_2F50AD0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v9; // rdx
  unsigned int v10; // eax
  __int64 v11; // r12
  __int64 v12; // rsi
  __int64 v13; // rcx
  __int16 v14; // r12
  __int64 v15; // rbx
  unsigned int v16; // r12d
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // r13
  __int64 v23; // r8
  __int64 v24; // rax
  const void *v25; // r9
  unsigned __int64 v26; // rdx
  size_t v27; // r10
  __int64 v28; // r13
  unsigned __int64 v29; // rcx
  __int64 v30; // r13
  __int64 *v31; // rdi
  __int64 *v32; // r12
  __int64 *v33; // rbx
  __int64 v34; // r14
  __int64 v35; // rax
  int v36; // r13d
  __int64 v37; // r13
  __int64 v38; // rdx
  __int64 v39; // r9
  unsigned __int64 v40; // rcx
  __int64 v41; // rax
  unsigned int v42; // edx
  int v43; // r10d
  unsigned __int64 v44; // r11
  _DWORD *v45; // rdx
  unsigned __int64 v46; // rcx
  unsigned int v47; // eax
  unsigned __int64 v48; // rcx
  unsigned __int64 v49; // r10
  int v50; // eax
  __int64 v51; // r9
  _DWORD *v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // rsi
  __int64 v55; // [rsp+8h] [rbp-B8h]
  unsigned int v56; // [rsp+10h] [rbp-B0h]
  __int64 v57; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v58; // [rsp+20h] [rbp-A0h]
  __int64 v59; // [rsp+20h] [rbp-A0h]
  const void *v60; // [rsp+28h] [rbp-98h]
  unsigned int v61; // [rsp+28h] [rbp-98h]
  int v62; // [rsp+28h] [rbp-98h]
  const void *v63; // [rsp+30h] [rbp-90h]
  __int64 v64; // [rsp+30h] [rbp-90h]
  unsigned int v65; // [rsp+30h] [rbp-90h]
  int v66; // [rsp+38h] [rbp-88h]
  int v67; // [rsp+3Ch] [rbp-84h]
  __int64 *v68; // [rsp+40h] [rbp-80h] BYREF
  __int64 v69; // [rsp+48h] [rbp-78h]
  _BYTE v70[112]; // [rsp+50h] [rbp-70h] BYREF

  v55 = a1 + 920;
  v9 = *(_QWORD *)(a1 + 920);
  v10 = *(_DWORD *)(a2 + 112) & 0x7FFFFFFF;
  v11 = 8LL * v10;
  v12 = v9 + v11;
  v67 = *(_DWORD *)(v9 + v11 + 4);
  if ( !v67 )
  {
    v47 = v10 + 1;
    v67 = *(_DWORD *)(a1 + 952);
    *(_DWORD *)(a1 + 952) = v67 + 1;
    v48 = *(unsigned int *)(a1 + 928);
    if ( v47 > (unsigned int)v48 )
    {
      v49 = v47;
      if ( v47 != v48 )
      {
        if ( v47 >= v48 )
        {
          v50 = *(_DWORD *)(a1 + 936);
          a5 = *(unsigned int *)(a1 + 940);
          v51 = v49 - v48;
          if ( v49 > *(unsigned int *)(a1 + 932) )
          {
            v59 = v49 - v48;
            v62 = *(_DWORD *)(a1 + 936);
            v65 = *(_DWORD *)(a1 + 940);
            sub_C8D5F0(v55, (const void *)(a1 + 936), v49, 8u, a5, v51);
            v9 = *(_QWORD *)(a1 + 920);
            v48 = *(unsigned int *)(a1 + 928);
            v51 = v59;
            v50 = v62;
            a5 = v65;
          }
          v52 = (_DWORD *)(v9 + 8 * v48);
          v53 = v51;
          do
          {
            if ( v52 )
            {
              *v52 = v50;
              v52[1] = a5;
            }
            v52 += 2;
            --v53;
          }
          while ( v53 );
          v54 = *(_QWORD *)(a1 + 920);
          *(_DWORD *)(a1 + 928) += v51;
          v12 = v11 + v54;
        }
        else
        {
          *(_DWORD *)(a1 + 928) = v47;
        }
      }
    }
    *(_DWORD *)(v12 + 4) = v67;
  }
  v13 = *(_QWORD *)(a1 + 8);
  v57 = a4;
  v68 = (__int64 *)v70;
  v69 = 0x800000000LL;
  v14 = *(_DWORD *)(*(_QWORD *)(v13 + 8) + 24LL * a3 + 16);
  v15 = *(_QWORD *)(v13 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v13 + 8) + 24LL * a3 + 16) >> 12);
  v16 = v14 & 0xFFF;
  while ( v15 )
  {
    v17 = sub_2E21610(*(_QWORD *)(a1 + 40), a2, v16);
    v22 = v17;
    if ( !*(_BYTE *)(v17 + 161) )
      sub_2E1AC90(v17, 0xFFFFFFFF, v18, v19, v20, v21);
    v23 = *(unsigned int *)(v22 + 120);
    v24 = (unsigned int)v69;
    v25 = *(const void **)(v22 + 112);
    v26 = v23 + (unsigned int)v69;
    v27 = 8 * v23;
    v28 = v23;
    if ( v26 > HIDWORD(v69) )
    {
      v60 = v25;
      v64 = 8 * v23;
      sub_C8D5F0((__int64)&v68, v70, v26, 8u, v23, (__int64)v25);
      v24 = (unsigned int)v69;
      v25 = v60;
      v27 = v64;
    }
    if ( v27 )
    {
      memcpy(&v68[v24], v25, v27);
      v24 = (unsigned int)v69;
    }
    a5 = v28 + v24;
    v15 += 2;
    LODWORD(v69) = v28 + v24;
    v29 = (unsigned int)(v28 + v24);
    v16 += *(__int16 *)(v15 - 2);
    if ( !*(_WORD *)(v15 - 2) )
    {
      v30 = v57;
      goto LABEL_12;
    }
  }
  v30 = v57;
  v29 = (unsigned int)v69;
LABEL_12:
  v31 = v68;
  v32 = &v68[v29];
  if ( v32 != v68 )
  {
    v33 = v68;
    v34 = v30;
    v63 = (const void *)(v30 + 16);
    do
    {
      v37 = *v33;
      v38 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 32LL);
      if ( *(_DWORD *)(v38 + 4LL * (*(_DWORD *)(*v33 + 112) & 0x7FFFFFFF)) )
      {
        sub_2E21040(*(_QWORD **)(a1 + 40), *v33, v38, v29, a5);
        v40 = *(unsigned int *)(a1 + 928);
        v41 = *(_DWORD *)(v37 + 112) & 0x7FFFFFFF;
        v42 = v41 + 1;
        if ( (int)v41 + 1 > (unsigned int)v40 && v42 != v40 )
        {
          if ( v42 >= v40 )
          {
            v39 = *(unsigned int *)(a1 + 936);
            v43 = *(_DWORD *)(a1 + 940);
            v44 = v42 - v40;
            if ( v42 > (unsigned __int64)*(unsigned int *)(a1 + 932) )
            {
              v66 = *(_DWORD *)(a1 + 940);
              v56 = *(_DWORD *)(a1 + 936);
              v58 = v42 - v40;
              v61 = *(_DWORD *)(v37 + 112) & 0x7FFFFFFF;
              sub_C8D5F0(v55, (const void *)(a1 + 936), v42, 8u, a1 + 936, v39);
              v40 = *(unsigned int *)(a1 + 928);
              v43 = v66;
              v39 = v56;
              v44 = v58;
              v41 = v61;
            }
            v45 = (_DWORD *)(*(_QWORD *)(a1 + 920) + 8 * v40);
            v46 = v44;
            do
            {
              if ( v45 )
              {
                *v45 = v39;
                v45[1] = v43;
              }
              v45 += 2;
              --v46;
            }
            while ( v46 );
            *(_DWORD *)(a1 + 928) += v44;
          }
          else
          {
            *(_DWORD *)(a1 + 928) = v42;
          }
        }
        *(_DWORD *)(*(_QWORD *)(a1 + 920) + 8 * v41 + 4) = v67;
        v35 = *(unsigned int *)(v34 + 8);
        v29 = *(unsigned int *)(v34 + 12);
        v36 = *(_DWORD *)(v37 + 112);
        if ( v35 + 1 > v29 )
        {
          sub_C8D5F0(v34, v63, v35 + 1, 4u, a5, v39);
          v35 = *(unsigned int *)(v34 + 8);
        }
        *(_DWORD *)(*(_QWORD *)v34 + 4 * v35) = v36;
        ++*(_DWORD *)(v34 + 8);
      }
      ++v33;
    }
    while ( v32 != v33 );
    v31 = v68;
  }
  if ( v31 != (__int64 *)v70 )
    _libc_free((unsigned __int64)v31);
}
