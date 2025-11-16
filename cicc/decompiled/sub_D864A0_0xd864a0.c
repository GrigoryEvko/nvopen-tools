// Function: sub_D864A0
// Address: 0xd864a0
//
__int64 __fastcall sub_D864A0(int *a1, __int64 a2)
{
  __int64 v3; // r14
  unsigned int v4; // eax
  unsigned int v5; // eax
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdi
  __m128i *v14; // rax
  __m128i *v15; // rcx
  __m128i *v16; // rdx
  __m128i *v17; // rax
  __m128i *v18; // rdx
  __int64 v19; // rax
  int v20; // eax
  __int64 v21; // rdi
  int *v22; // r13
  __int64 v23; // rbx
  __int64 v24; // r12
  unsigned int v25; // eax
  unsigned int v26; // eax
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdi
  __m128i *v35; // rax
  __m128i *v36; // rcx
  __m128i *v37; // rdx
  __m128i *v38; // rax
  __m128i *v39; // rdx
  __int64 v40; // rax
  int v41; // eax
  __int64 v42; // rdi
  unsigned int v44; // eax
  unsigned int v45; // eax

  v3 = sub_22077B0(168);
  *(_QWORD *)(v3 + 32) = *((_QWORD *)a1 + 4);
  v4 = a1[12];
  *(_DWORD *)(v3 + 48) = v4;
  if ( v4 > 0x40 )
  {
    sub_C43780(v3 + 40, (const void **)a1 + 5);
    v45 = a1[16];
    *(_DWORD *)(v3 + 64) = v45;
    if ( v45 <= 0x40 )
      goto LABEL_3;
  }
  else
  {
    *(_QWORD *)(v3 + 40) = *((_QWORD *)a1 + 5);
    v5 = a1[16];
    *(_DWORD *)(v3 + 64) = v5;
    if ( v5 <= 0x40 )
    {
LABEL_3:
      *(_QWORD *)(v3 + 56) = *((_QWORD *)a1 + 7);
      goto LABEL_4;
    }
  }
  sub_C43780(v3 + 56, (const void **)a1 + 7);
LABEL_4:
  v6 = *((_QWORD *)a1 + 11);
  *(_DWORD *)(v3 + 80) = 0;
  *(_QWORD *)(v3 + 88) = 0;
  *(_QWORD *)(v3 + 96) = v3 + 80;
  *(_QWORD *)(v3 + 104) = v3 + 80;
  *(_QWORD *)(v3 + 112) = 0;
  if ( v6 )
  {
    v7 = sub_D85910(v6, v3 + 80);
    v8 = v7;
    do
    {
      v9 = v7;
      v7 = *(_QWORD *)(v7 + 16);
    }
    while ( v7 );
    *(_QWORD *)(v3 + 96) = v9;
    v10 = v8;
    do
    {
      v11 = v10;
      v10 = *(_QWORD *)(v10 + 24);
    }
    while ( v10 );
    v12 = *((_QWORD *)a1 + 14);
    *(_QWORD *)(v3 + 104) = v11;
    *(_QWORD *)(v3 + 88) = v8;
    *(_QWORD *)(v3 + 112) = v12;
  }
  v13 = *((_QWORD *)a1 + 17);
  *(_DWORD *)(v3 + 128) = 0;
  *(_QWORD *)(v3 + 136) = 0;
  *(_QWORD *)(v3 + 144) = v3 + 128;
  *(_QWORD *)(v3 + 152) = v3 + 128;
  *(_QWORD *)(v3 + 160) = 0;
  if ( v13 )
  {
    v14 = sub_D85C90(v13, v3 + 128);
    v15 = v14;
    do
    {
      v16 = v14;
      v14 = (__m128i *)v14[1].m128i_i64[0];
    }
    while ( v14 );
    *(_QWORD *)(v3 + 144) = v16;
    v17 = v15;
    do
    {
      v18 = v17;
      v17 = (__m128i *)v17[1].m128i_i64[1];
    }
    while ( v17 );
    v19 = *((_QWORD *)a1 + 20);
    *(_QWORD *)(v3 + 152) = v18;
    *(_QWORD *)(v3 + 136) = v15;
    *(_QWORD *)(v3 + 160) = v19;
  }
  v20 = *a1;
  v21 = *((_QWORD *)a1 + 3);
  *(_QWORD *)(v3 + 8) = a2;
  *(_QWORD *)(v3 + 16) = 0;
  *(_DWORD *)v3 = v20;
  *(_QWORD *)(v3 + 24) = 0;
  if ( v21 )
    *(_QWORD *)(v3 + 24) = sub_D864A0(v21, v3);
  v22 = (int *)*((_QWORD *)a1 + 2);
  if ( v22 )
  {
    v23 = v3;
    do
    {
      v24 = v23;
      v23 = sub_22077B0(168);
      *(_QWORD *)(v23 + 32) = *((_QWORD *)v22 + 4);
      v25 = v22[12];
      *(_DWORD *)(v23 + 48) = v25;
      if ( v25 > 0x40 )
      {
        sub_C43780(v23 + 40, (const void **)v22 + 5);
        v44 = v22[16];
        *(_DWORD *)(v23 + 64) = v44;
        if ( v44 > 0x40 )
        {
LABEL_40:
          sub_C43780(v23 + 56, (const void **)v22 + 7);
          goto LABEL_23;
        }
      }
      else
      {
        *(_QWORD *)(v23 + 40) = *((_QWORD *)v22 + 5);
        v26 = v22[16];
        *(_DWORD *)(v23 + 64) = v26;
        if ( v26 > 0x40 )
          goto LABEL_40;
      }
      *(_QWORD *)(v23 + 56) = *((_QWORD *)v22 + 7);
LABEL_23:
      *(_DWORD *)(v23 + 80) = 0;
      *(_QWORD *)(v23 + 88) = 0;
      *(_QWORD *)(v23 + 96) = v23 + 80;
      *(_QWORD *)(v23 + 104) = v23 + 80;
      *(_QWORD *)(v23 + 112) = 0;
      v27 = *((_QWORD *)v22 + 11);
      if ( v27 )
      {
        v28 = sub_D85910(v27, v23 + 80);
        v29 = v28;
        do
        {
          v30 = v28;
          v28 = *(_QWORD *)(v28 + 16);
        }
        while ( v28 );
        *(_QWORD *)(v23 + 96) = v30;
        v31 = v29;
        do
        {
          v32 = v31;
          v31 = *(_QWORD *)(v31 + 24);
        }
        while ( v31 );
        *(_QWORD *)(v23 + 104) = v32;
        v33 = *((_QWORD *)v22 + 14);
        *(_QWORD *)(v23 + 88) = v29;
        *(_QWORD *)(v23 + 112) = v33;
      }
      *(_DWORD *)(v23 + 128) = 0;
      *(_QWORD *)(v23 + 136) = 0;
      *(_QWORD *)(v23 + 144) = v23 + 128;
      *(_QWORD *)(v23 + 152) = v23 + 128;
      *(_QWORD *)(v23 + 160) = 0;
      v34 = *((_QWORD *)v22 + 17);
      if ( v34 )
      {
        v35 = sub_D85C90(v34, v23 + 128);
        v36 = v35;
        do
        {
          v37 = v35;
          v35 = (__m128i *)v35[1].m128i_i64[0];
        }
        while ( v35 );
        *(_QWORD *)(v23 + 144) = v37;
        v38 = v36;
        do
        {
          v39 = v38;
          v38 = (__m128i *)v38[1].m128i_i64[1];
        }
        while ( v38 );
        *(_QWORD *)(v23 + 152) = v39;
        v40 = *((_QWORD *)v22 + 20);
        *(_QWORD *)(v23 + 136) = v36;
        *(_QWORD *)(v23 + 160) = v40;
      }
      v41 = *v22;
      *(_QWORD *)(v23 + 16) = 0;
      *(_QWORD *)(v23 + 24) = 0;
      *(_DWORD *)v23 = v41;
      *(_QWORD *)(v24 + 16) = v23;
      *(_QWORD *)(v23 + 8) = v24;
      v42 = *((_QWORD *)v22 + 3);
      if ( v42 )
        *(_QWORD *)(v23 + 24) = sub_D864A0(v42, v23);
      v22 = (int *)*((_QWORD *)v22 + 2);
    }
    while ( v22 );
  }
  return v3;
}
