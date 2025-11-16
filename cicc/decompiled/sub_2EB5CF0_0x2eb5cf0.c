// Function: sub_2EB5CF0
// Address: 0x2eb5cf0
//
void __fastcall sub_2EB5CF0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  unsigned __int64 v7; // rdx
  __int64 v8; // rbx
  unsigned int v9; // r14d
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  _QWORD *v14; // r15
  __int64 v15; // rax
  unsigned int v16; // r13d
  _QWORD *v17; // rbx
  __int64 v18; // r15
  unsigned int *v19; // r10
  unsigned int *v20; // r14
  _QWORD *v21; // rax
  __int64 v22; // r12
  unsigned int v23; // ebx
  _QWORD *v24; // r13
  __int64 v25; // r15
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  unsigned int v29; // eax
  _QWORD *i; // rax
  __int64 v31; // r14
  __int64 v32; // rdx
  __int64 v33; // r15
  unsigned int v34; // r12d
  __int64 v35; // rax
  __int64 v36; // [rsp+8h] [rbp-1C8h]
  unsigned int v37; // [rsp+1Ch] [rbp-1B4h]
  __int64 v38; // [rsp+20h] [rbp-1B0h]
  unsigned int *v39; // [rsp+38h] [rbp-198h]
  __int64 v40; // [rsp+38h] [rbp-198h]
  _QWORD *v41; // [rsp+40h] [rbp-190h] BYREF
  __int64 v42; // [rsp+48h] [rbp-188h]
  _QWORD v43[8]; // [rsp+50h] [rbp-180h] BYREF
  _BYTE *v44; // [rsp+90h] [rbp-140h] BYREF
  __int64 v45; // [rsp+98h] [rbp-138h]
  _BYTE v46[304]; // [rsp+A0h] [rbp-130h] BYREF

  v6 = a1;
  v7 = *(unsigned int *)(a1 + 8);
  v41 = v43;
  v37 = v7;
  v42 = 0x800000001LL;
  v43[0] = 0;
  if ( (unsigned int)v7 > 8 )
  {
    sub_C8D5F0((__int64)&v41, v43, v7, 8u, a5, a6);
  }
  else if ( (unsigned int)v7 <= 1 )
  {
    v44 = v46;
    v45 = 0x2000000000LL;
    v15 = (unsigned int)(v7 - 1);
    if ( (unsigned int)v15 <= 1 )
      return;
    v14 = v43;
    goto LABEL_8;
  }
  v8 = 8;
  v9 = 1;
  do
  {
    v10 = sub_2EB5B40(a1, *(_QWORD *)(*(_QWORD *)a1 + v8), v7, a4, a5, a6);
    a4 = HIDWORD(v42);
    v11 = v10;
    *(_QWORD *)(v10 + 16) = *(_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(v10 + 4));
    v12 = (unsigned int)v42;
    v13 = (unsigned int)v42 + 1LL;
    if ( v13 > a4 )
    {
      sub_C8D5F0((__int64)&v41, v43, v13, 8u, a5, a6);
      v12 = (unsigned int)v42;
    }
    v7 = (unsigned __int64)v41;
    ++v9;
    v8 += 8;
    v41[v12] = v11;
    LODWORD(v42) = v42 + 1;
  }
  while ( v9 < v37 );
  v6 = a1;
  v14 = v41;
  v44 = v46;
  v45 = 0x2000000000LL;
  v15 = v37 - 1;
  if ( (unsigned int)v15 > 1 )
  {
LABEL_8:
    v16 = v37;
    v36 = v6;
    v17 = v14;
    v38 = v15;
    while ( 1 )
    {
      v18 = v17[v38];
      v19 = *(unsigned int **)(v18 + 24);
      *(_DWORD *)(v18 + 8) = *(_DWORD *)(v18 + 4);
      v20 = v19;
      v39 = &v19[*(unsigned int *)(v18 + 32)];
      if ( v39 != v19 )
      {
        v21 = v17;
        v22 = v18;
        v23 = v16;
        v24 = v21;
        do
        {
          v25 = v24[*v20];
          v26 = (unsigned int)v45;
          if ( *(_DWORD *)(v25 + 4) < v23 )
          {
            v28 = *(unsigned int *)(v25 + 12);
          }
          else
          {
            do
            {
              if ( v26 + 1 > (unsigned __int64)HIDWORD(v45) )
              {
                sub_C8D5F0((__int64)&v44, v46, v26 + 1, 8u, a5, a6);
                v26 = (unsigned int)v45;
              }
              *(_QWORD *)&v44[8 * v26] = v25;
              v26 = (unsigned int)(v45 + 1);
              LODWORD(v45) = v45 + 1;
              v25 = v24[*(unsigned int *)(v25 + 4)];
            }
            while ( *(_DWORD *)(v25 + 4) >= v23 );
            a4 = v24[*(unsigned int *)(v25 + 12)];
            do
            {
              while ( 1 )
              {
                v27 = v25;
                v25 = *(_QWORD *)&v44[8 * (unsigned int)v26 - 8];
                LODWORD(v45) = v26 - 1;
                *(_DWORD *)(v25 + 4) = *(_DWORD *)(v27 + 4);
                if ( *(_DWORD *)(a4 + 8) >= *(_DWORD *)(v24[*(unsigned int *)(v25 + 12)] + 8LL) )
                  break;
                *(_DWORD *)(v25 + 12) = *(_DWORD *)(v27 + 12);
                LODWORD(v26) = v45;
                if ( !(_DWORD)v45 )
                  goto LABEL_19;
              }
              a4 = v24[*(unsigned int *)(v25 + 12)];
              LODWORD(v26) = v45;
            }
            while ( (_DWORD)v45 );
LABEL_19:
            v28 = *(unsigned int *)(v25 + 12);
            v24 = v41;
          }
          v29 = *(_DWORD *)(v24[v28] + 8LL);
          if ( *(_DWORD *)(v22 + 8) > v29 )
            *(_DWORD *)(v22 + 8) = v29;
          ++v20;
        }
        while ( v39 != v20 );
        v16 = v23;
      }
      --v16;
      --v38;
      if ( v16 == 2 )
        break;
      v17 = v41;
    }
    if ( v37 > 2 )
    {
      v40 = 2;
      for ( i = v41; ; i = v41 )
      {
        v31 = i[v40];
        v32 = *(unsigned int *)(v31 + 8);
        v33 = *(_QWORD *)(v31 + 16);
        v34 = *(_DWORD *)i[v32];
        while ( 1 )
        {
          v35 = sub_2EB5B40(v36, v33, v32, a4, a5, a6);
          if ( *(_DWORD *)v35 <= v34 )
            break;
          v33 = *(_QWORD *)(v35 + 16);
        }
        ++v40;
        ++v16;
        *(_QWORD *)(v31 + 16) = v33;
        if ( v37 <= v16 )
          break;
      }
    }
    if ( v44 != v46 )
      _libc_free((unsigned __int64)v44);
    v14 = v41;
  }
  if ( v14 != v43 )
    _libc_free((unsigned __int64)v14);
}
