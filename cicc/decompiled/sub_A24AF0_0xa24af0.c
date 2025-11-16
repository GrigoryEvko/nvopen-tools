// Function: sub_A24AF0
// Address: 0xa24af0
//
void __fastcall sub_A24AF0(__int64 *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  _QWORD *v7; // rsi
  _QWORD *v8; // r12
  unsigned int v9; // r13d
  unsigned int v10; // ecx
  unsigned int v11; // edx
  unsigned int v12; // r14d
  int v13; // eax
  _QWORD *v14; // r12
  __int64 v15; // rdx
  unsigned int v16; // eax
  unsigned int v17; // ecx
  int v18; // r13d
  _BYTE *v19; // rsi
  _QWORD *v20; // rbx
  __int64 v21; // rdi
  __int64 v22; // r12
  __int64 v23; // rax
  size_t v24; // rdx
  size_t v25; // r12
  _BYTE *v26; // r15
  unsigned __int64 v27; // rdx
  _QWORD *v28; // r15
  __int64 v29; // rax
  unsigned int v30; // edx
  __int64 v31; // rbx
  __int64 v32; // r14
  __int64 v33; // r13
  __int64 v34; // rax
  _QWORD *v35; // rax
  __int64 v36; // r13
  __int64 v37; // rax
  _QWORD *v38; // rax
  __int64 v39; // r13
  __int64 v40; // rax
  _QWORD *v41; // rax
  __int64 v42; // r13
  __int64 v43; // rax
  _QWORD *v44; // rax
  __int64 v45; // rdi
  __int64 v46; // rax
  volatile signed __int32 *v47; // rax
  unsigned int v48; // r13d
  _QWORD *v49; // rbx
  int v50; // r12d
  __int64 v51; // rax
  _QWORD *v55; // [rsp+30h] [rbp-210h]
  _QWORD *v56; // [rsp+38h] [rbp-208h]
  __int64 v57; // [rsp+40h] [rbp-200h] BYREF
  volatile signed __int32 *v58; // [rsp+48h] [rbp-1F8h]
  _QWORD *v59; // [rsp+50h] [rbp-1F0h] BYREF
  volatile signed __int32 *v60; // [rsp+58h] [rbp-1E8h]
  __int64 v61; // [rsp+60h] [rbp-1E0h]
  _QWORD v62[3]; // [rsp+68h] [rbp-1D8h] BYREF
  __int64 v63; // [rsp+80h] [rbp-1C0h]
  __int64 v64; // [rsp+88h] [rbp-1B8h]
  __int64 v65; // [rsp+90h] [rbp-1B0h]
  __int64 v66; // [rsp+98h] [rbp-1A8h]
  __int64 v67; // [rsp+A0h] [rbp-1A0h]
  char v68; // [rsp+B0h] [rbp-190h]
  __int64 v69; // [rsp+B8h] [rbp-188h]
  __int64 v70; // [rsp+C0h] [rbp-180h]
  __int64 v71; // [rsp+C8h] [rbp-178h]
  __int64 v72; // [rsp+D0h] [rbp-170h]
  __int64 v73; // [rsp+D8h] [rbp-168h]
  __int64 v74; // [rsp+E0h] [rbp-160h]
  _BYTE *v75; // [rsp+F0h] [rbp-150h] BYREF
  _BYTE *v76; // [rsp+F8h] [rbp-148h]
  unsigned __int64 v77; // [rsp+100h] [rbp-140h]
  _BYTE v78[312]; // [rsp+108h] [rbp-138h] BYREF

  if ( a3 )
  {
    sub_A188E0(a4, 35);
    sub_A188E0(a4, a3);
    v7 = &v75;
    v76 = 0;
    v75 = v78;
    v59 = v62;
    v77 = 256;
    v60 = 0;
    v61 = 0;
    v62[0] = &v75;
    v62[1] = 0;
    v62[2] = 0;
    v63 = 0;
    v64 = 2;
    v65 = 0;
    v66 = 0;
    v67 = 0;
    v68 = 0;
    v69 = 0;
    v70 = 0;
    v71 = 0;
    v72 = 0;
    v73 = 0;
    v74 = 0;
    v55 = &a2[a3];
    if ( v55 == a2 )
    {
      sub_A173F0((__int64)&v59, &v75);
      sub_A188E0(a4, (__int64)v76);
      v22 = (__int64)v76;
    }
    else
    {
      v8 = a2;
      do
      {
        sub_B91420(*v8, v7);
        v9 = HIDWORD(v63);
        v10 = v63;
        v12 = v11;
        if ( v11 > 0x1F )
        {
          v56 = v8;
          do
          {
            v13 = (v12 & 0x1F | 0x20) << v10;
            v10 += 6;
            v9 |= v13;
            HIDWORD(v63) = v9;
            if ( v10 > 0x1F )
            {
              v14 = (_QWORD *)v62[0];
              v15 = *(_QWORD *)(v62[0] + 8LL);
              if ( (unsigned __int64)(v15 + 4) > *(_QWORD *)(v62[0] + 16LL) )
              {
                v7 = (_QWORD *)(v62[0] + 24LL);
                sub_C8D290(v62[0], v62[0] + 24LL, v15 + 4, 1);
                v15 = v14[1];
              }
              *(_DWORD *)(*v14 + v15) = v9;
              v9 = 0;
              v14[1] += 4LL;
              if ( (_DWORD)v63 )
                v9 = (v12 & 0x1F | 0x20) >> (32 - v63);
              v10 = ((_BYTE)v63 + 6) & 0x1F;
            }
            v12 >>= 5;
            LODWORD(v63) = v10;
          }
          while ( v12 > 0x1F );
          v8 = v56;
        }
        v16 = v12 << v10;
        v17 = v10 + 6;
        v18 = v16 | v9;
        HIDWORD(v63) = v18;
        if ( v17 > 0x1F )
        {
          v28 = (_QWORD *)v62[0];
          v29 = *(_QWORD *)(v62[0] + 8LL);
          if ( (unsigned __int64)(v29 + 4) > *(_QWORD *)(v62[0] + 16LL) )
          {
            v7 = (_QWORD *)(v62[0] + 24LL);
            sub_C8D290(v62[0], v62[0] + 24LL, v29 + 4, 1);
            v29 = v28[1];
          }
          *(_DWORD *)(*v28 + v29) = v18;
          v30 = 0;
          v28[1] += 4LL;
          if ( (_DWORD)v63 )
            v30 = v12 >> (32 - v63);
          HIDWORD(v63) = v30;
          LODWORD(v63) = ((_BYTE)v63 + 6) & 0x1F;
        }
        else
        {
          LODWORD(v63) = v17;
        }
        ++v8;
      }
      while ( v55 != v8 );
      if ( (_DWORD)v63 )
      {
        v49 = (_QWORD *)v62[0];
        v50 = HIDWORD(v63);
        v51 = *(_QWORD *)(v62[0] + 8LL);
        if ( (unsigned __int64)(v51 + 4) > *(_QWORD *)(v62[0] + 16LL) )
        {
          v7 = (_QWORD *)(v62[0] + 24LL);
          sub_C8D290(v62[0], v62[0] + 24LL, v51 + 4, 1);
          v51 = v49[1];
        }
        *(_DWORD *)(*v49 + v51) = v50;
        v49[1] += 4LL;
        v63 = 0;
      }
      sub_A173F0((__int64)&v59, v7);
      v19 = v76;
      sub_A188E0(a4, (__int64)v76);
      v20 = a2;
      do
      {
        v23 = sub_B91420(*v20, v19);
        v21 = (__int64)v76;
        v25 = v24;
        v26 = (_BYTE *)v23;
        v27 = (unsigned __int64)&v76[v24];
        if ( v27 > v77 )
        {
          v19 = v78;
          sub_C8D290(&v75, v78, v27, 1);
          v21 = (__int64)v76;
        }
        if ( v25 )
        {
          v19 = v26;
          memcpy(&v75[v21], v26, v25);
          v21 = (__int64)v76;
        }
        v22 = v21 + v25;
        ++v20;
        v76 = (_BYTE *)v22;
      }
      while ( v55 != v20 );
    }
    v31 = (__int64)v75;
    v32 = *a1;
    sub_A23770(&v57);
    v33 = v57;
    v34 = *(unsigned int *)(v57 + 8);
    if ( v34 + 1 > (unsigned __int64)*(unsigned int *)(v57 + 12) )
    {
      sub_C8D5F0(v57, v57 + 16, v34 + 1, 16);
      v34 = *(unsigned int *)(v33 + 8);
    }
    v35 = (_QWORD *)(*(_QWORD *)v33 + 16 * v34);
    *v35 = 35;
    v35[1] = 1;
    ++*(_DWORD *)(v33 + 8);
    v36 = v57;
    v37 = *(unsigned int *)(v57 + 8);
    if ( v37 + 1 > (unsigned __int64)*(unsigned int *)(v57 + 12) )
    {
      sub_C8D5F0(v57, v57 + 16, v37 + 1, 16);
      v37 = *(unsigned int *)(v36 + 8);
    }
    v38 = (_QWORD *)(*(_QWORD *)v36 + 16 * v37);
    *v38 = 6;
    v38[1] = 4;
    ++*(_DWORD *)(v36 + 8);
    v39 = v57;
    v40 = *(unsigned int *)(v57 + 8);
    if ( v40 + 1 > (unsigned __int64)*(unsigned int *)(v57 + 12) )
    {
      sub_C8D5F0(v57, v57 + 16, v40 + 1, 16);
      v40 = *(unsigned int *)(v39 + 8);
    }
    v41 = (_QWORD *)(*(_QWORD *)v39 + 16 * v40);
    *v41 = 6;
    v41[1] = 4;
    ++*(_DWORD *)(v39 + 8);
    v42 = v57;
    v43 = *(unsigned int *)(v57 + 8);
    if ( v43 + 1 > (unsigned __int64)*(unsigned int *)(v57 + 12) )
    {
      sub_C8D5F0(v57, v57 + 16, v43 + 1, 16);
      v43 = *(unsigned int *)(v42 + 8);
    }
    v44 = (_QWORD *)(*(_QWORD *)v42 + 16 * v43);
    *v44 = 0;
    v44[1] = 10;
    ++*(_DWORD *)(v42 + 8);
    v45 = *a1;
    v46 = v57;
    v57 = 0;
    v59 = (_QWORD *)v46;
    v47 = v58;
    v58 = 0;
    v60 = v47;
    v48 = sub_A1AB30(v45, (__int64 *)&v59);
    if ( v60 )
      sub_A191D0(v60);
    if ( v58 )
      sub_A191D0(v58);
    sub_A1B020(v32, v48, *(_QWORD *)a4, *(unsigned int *)(a4 + 8), v31, v22, (unsigned int)v59, 0);
    *(_DWORD *)(a4 + 8) = 0;
    if ( v75 != v78 )
      _libc_free(v75, v48);
  }
}
