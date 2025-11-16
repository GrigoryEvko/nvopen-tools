// Function: sub_D86C20
// Address: 0xd86c20
//
__int64 __fastcall sub_D86C20(int *a1, __int64 a2)
{
  __int64 v3; // rax
  int *v4; // rdi
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  int *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  int v20; // eax
  __int64 v21; // rdi
  int v22; // eax
  int *v23; // r13
  __int64 v24; // rbx
  __int64 v25; // r12
  __int64 v26; // rax
  int *v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  int *v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rcx
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rax
  int v41; // eax
  __int64 v42; // rdi

  v3 = sub_22077B0(144);
  v4 = (int *)*((_QWORD *)a1 + 7);
  v5 = v3;
  v6 = *((_QWORD *)a1 + 4);
  *(_DWORD *)(v5 + 48) = 0;
  *(_QWORD *)(v5 + 32) = v6;
  *(_QWORD *)(v5 + 56) = 0;
  *(_QWORD *)(v5 + 64) = v5 + 48;
  *(_QWORD *)(v5 + 72) = v5 + 48;
  *(_QWORD *)(v5 + 80) = 0;
  if ( v4 )
  {
    v7 = sub_D864A0(v4, v5 + 48);
    v8 = v7;
    do
    {
      v9 = v7;
      v7 = *(_QWORD *)(v7 + 16);
    }
    while ( v7 );
    *(_QWORD *)(v5 + 64) = v9;
    v10 = v8;
    do
    {
      v11 = v10;
      v10 = *(_QWORD *)(v10 + 24);
    }
    while ( v10 );
    v12 = *((_QWORD *)a1 + 10);
    *(_QWORD *)(v5 + 72) = v11;
    *(_QWORD *)(v5 + 56) = v8;
    *(_QWORD *)(v5 + 80) = v12;
  }
  v13 = (int *)*((_QWORD *)a1 + 13);
  *(_DWORD *)(v5 + 96) = 0;
  *(_QWORD *)(v5 + 104) = 0;
  *(_QWORD *)(v5 + 112) = v5 + 96;
  *(_QWORD *)(v5 + 120) = v5 + 96;
  *(_QWORD *)(v5 + 128) = 0;
  if ( v13 )
  {
    v14 = sub_D86860(v13, v5 + 96);
    v15 = v14;
    do
    {
      v16 = v14;
      v14 = *(_QWORD *)(v14 + 16);
    }
    while ( v14 );
    *(_QWORD *)(v5 + 112) = v16;
    v17 = v15;
    do
    {
      v18 = v17;
      v17 = *(_QWORD *)(v17 + 24);
    }
    while ( v17 );
    v19 = *((_QWORD *)a1 + 16);
    *(_QWORD *)(v5 + 120) = v18;
    *(_QWORD *)(v5 + 104) = v15;
    *(_QWORD *)(v5 + 128) = v19;
  }
  v20 = a1[34];
  v21 = *((_QWORD *)a1 + 3);
  *(_QWORD *)(v5 + 8) = a2;
  *(_QWORD *)(v5 + 16) = 0;
  *(_DWORD *)(v5 + 136) = v20;
  v22 = *a1;
  *(_QWORD *)(v5 + 24) = 0;
  *(_DWORD *)v5 = v22;
  if ( v21 )
    *(_QWORD *)(v5 + 24) = sub_D86C20(v21, v5);
  v23 = (int *)*((_QWORD *)a1 + 2);
  if ( v23 )
  {
    v24 = v5;
    do
    {
      v25 = v24;
      v24 = sub_22077B0(144);
      v26 = *((_QWORD *)v23 + 4);
      *(_DWORD *)(v24 + 48) = 0;
      *(_QWORD *)(v24 + 32) = v26;
      *(_QWORD *)(v24 + 56) = 0;
      *(_QWORD *)(v24 + 64) = v24 + 48;
      *(_QWORD *)(v24 + 72) = v24 + 48;
      *(_QWORD *)(v24 + 80) = 0;
      v27 = (int *)*((_QWORD *)v23 + 7);
      if ( v27 )
      {
        v28 = sub_D864A0(v27, v24 + 48);
        v29 = v28;
        do
        {
          v30 = v28;
          v28 = *(_QWORD *)(v28 + 16);
        }
        while ( v28 );
        *(_QWORD *)(v24 + 64) = v30;
        v31 = v29;
        do
        {
          v32 = v31;
          v31 = *(_QWORD *)(v31 + 24);
        }
        while ( v31 );
        *(_QWORD *)(v24 + 72) = v32;
        v33 = *((_QWORD *)v23 + 10);
        *(_QWORD *)(v24 + 56) = v29;
        *(_QWORD *)(v24 + 80) = v33;
      }
      *(_DWORD *)(v24 + 96) = 0;
      *(_QWORD *)(v24 + 104) = 0;
      *(_QWORD *)(v24 + 112) = v24 + 96;
      *(_QWORD *)(v24 + 120) = v24 + 96;
      *(_QWORD *)(v24 + 128) = 0;
      v34 = (int *)*((_QWORD *)v23 + 13);
      if ( v34 )
      {
        v35 = sub_D86860(v34, v24 + 96);
        v36 = v35;
        do
        {
          v37 = v35;
          v35 = *(_QWORD *)(v35 + 16);
        }
        while ( v35 );
        *(_QWORD *)(v24 + 112) = v37;
        v38 = v36;
        do
        {
          v39 = v38;
          v38 = *(_QWORD *)(v38 + 24);
        }
        while ( v38 );
        *(_QWORD *)(v24 + 120) = v39;
        v40 = *((_QWORD *)v23 + 16);
        *(_QWORD *)(v24 + 104) = v36;
        *(_QWORD *)(v24 + 128) = v40;
      }
      *(_DWORD *)(v24 + 136) = v23[34];
      v41 = *v23;
      *(_QWORD *)(v24 + 16) = 0;
      *(_DWORD *)v24 = v41;
      *(_QWORD *)(v24 + 24) = 0;
      *(_QWORD *)(v25 + 16) = v24;
      *(_QWORD *)(v24 + 8) = v25;
      v42 = *((_QWORD *)v23 + 3);
      if ( v42 )
        *(_QWORD *)(v24 + 24) = sub_D86C20(v42, v24);
      v23 = (int *)*((_QWORD *)v23 + 2);
    }
    while ( v23 );
  }
  return v5;
}
