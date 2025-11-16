// Function: sub_15253D0
// Address: 0x15253d0
//
void __fastcall sub_15253D0(_DWORD *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r13
  int v5; // ecx
  __int64 v6; // r9
  unsigned int v7; // r15d
  unsigned int v8; // eax
  unsigned int v9; // ecx
  int v10; // eax
  unsigned __int8 v11; // al
  unsigned __int64 v12; // r9
  unsigned int v13; // eax
  unsigned int v14; // r15d
  unsigned int v15; // ecx
  int v16; // eax
  unsigned int v17; // r10d
  unsigned int v18; // r10d
  int v19; // eax
  __int64 v20; // rax
  __int64 v21; // rdx
  int v22; // ecx
  unsigned int v23; // r15d
  unsigned int v24; // eax
  unsigned int v25; // ecx
  int v26; // eax
  unsigned __int8 v27; // al
  __int64 v28; // rdi
  __int64 v29; // rdx
  unsigned int v30; // edx
  int v31; // eax
  unsigned int v32; // r15d
  int v33; // edx
  unsigned int v34; // ecx
  int v35; // r15d
  __int64 v36; // rdi
  __int64 v37; // rdx
  unsigned int v38; // eax
  int v39; // edx
  unsigned int v40; // r15d
  __int64 v41; // rdi
  __int64 v42; // rdx
  int v43; // edx
  unsigned int v44; // eax
  unsigned int v45; // ecx
  __int64 v46; // [rsp+0h] [rbp-50h]
  unsigned __int64 v47; // [rsp+8h] [rbp-48h]
  int v48; // [rsp+8h] [rbp-48h]
  int v49; // [rsp+8h] [rbp-48h]
  unsigned int v50; // [rsp+10h] [rbp-40h]
  __int64 v51; // [rsp+10h] [rbp-40h]
  __int64 v52; // [rsp+10h] [rbp-40h]
  unsigned int v53; // [rsp+10h] [rbp-40h]
  __int64 v54; // [rsp+18h] [rbp-38h]

  sub_1524D80(a1, 2u, a1[4]);
  sub_1524E40(a1, *(_DWORD *)(a2 + 8), 5);
  v3 = *(unsigned int *)(a2 + 8);
  v46 = 16 * v3;
  if ( (_DWORD)v3 )
  {
    v4 = 0;
    do
    {
      v5 = a1[2];
      v6 = v4 + *(_QWORD *)a2;
      v7 = *(_BYTE *)(v6 + 8) & 1;
      v8 = v7 << v5;
      v9 = v5 + 1;
      v10 = a1[3] | v8;
      a1[3] = v10;
      if ( v9 > 0x1F )
      {
        v28 = *(_QWORD *)a1;
        v29 = *(unsigned int *)(*(_QWORD *)a1 + 8LL);
        if ( (unsigned __int64)*(unsigned int *)(*(_QWORD *)a1 + 12LL) - v29 <= 3 )
        {
          v48 = v10;
          v51 = v6;
          sub_16CD150(v28, v28 + 16, v29 + 4, 1);
          v10 = v48;
          v6 = v51;
          v29 = *(unsigned int *)(v28 + 8);
        }
        *(_DWORD *)(*(_QWORD *)v28 + v29) = v10;
        v30 = 0;
        *(_DWORD *)(v28 + 8) += 4;
        v31 = a1[2];
        v32 = v7 >> (32 - v31);
        if ( v31 )
          v30 = v32;
        a1[3] = v30;
        a1[2] = ((_BYTE)v31 + 1) & 0x1F;
      }
      else
      {
        a1[2] = v9;
      }
      v11 = *(_BYTE *)(v6 + 8);
      if ( (v11 & 1) != 0 )
      {
        v12 = *(_QWORD *)v6;
        v13 = v12;
        if ( v12 == (unsigned int)v12 )
        {
          sub_1524E40(a1, v12, 8);
        }
        else
        {
          v14 = a1[3];
          v15 = a1[2];
          if ( v12 > 0x7F )
          {
            do
            {
              v18 = v12 & 0x7F | 0x80;
              v19 = v18 << v15;
              v15 += 8;
              v14 |= v19;
              a1[3] = v14;
              if ( v15 > 0x1F )
              {
                v20 = *(_QWORD *)a1;
                v21 = *(unsigned int *)(*(_QWORD *)a1 + 8LL);
                if ( (unsigned __int64)*(unsigned int *)(*(_QWORD *)a1 + 12LL) - v21 <= 3 )
                {
                  v47 = v12;
                  v50 = v12 & 0x7F | 0x80;
                  v54 = *(_QWORD *)a1;
                  sub_16CD150(*(_QWORD *)a1, v20 + 16, v21 + 4, 1);
                  v20 = v54;
                  v12 = v47;
                  v18 = v50;
                  v21 = *(unsigned int *)(v54 + 8);
                }
                *(_DWORD *)(*(_QWORD *)v20 + v21) = v14;
                v14 = 0;
                *(_DWORD *)(v20 + 8) += 4;
                v16 = a1[2];
                v17 = v18 >> (32 - v16);
                if ( v16 )
                  v14 = v17;
                v15 = ((_BYTE)v16 + 8) & 0x1F;
                a1[3] = v14;
              }
              v12 >>= 7;
              a1[2] = v15;
            }
            while ( v12 > 0x7F );
            v13 = v12;
          }
          v33 = v13 << v15;
          v34 = v15 + 8;
          v35 = v33 | v14;
          a1[3] = v35;
          if ( v34 > 0x1F )
          {
            v41 = *(_QWORD *)a1;
            v42 = *(unsigned int *)(*(_QWORD *)a1 + 8LL);
            if ( (unsigned __int64)*(unsigned int *)(*(_QWORD *)a1 + 12LL) - v42 <= 3 )
            {
              v53 = v13;
              sub_16CD150(v41, v41 + 16, v42 + 4, 1);
              v13 = v53;
              v42 = *(unsigned int *)(v41 + 8);
            }
            *(_DWORD *)(*(_QWORD *)v41 + v42) = v35;
            *(_DWORD *)(v41 + 8) += 4;
            v43 = a1[2];
            v44 = v13 >> (32 - v43);
            v45 = 0;
            if ( v43 )
              v45 = v44;
            a1[3] = v45;
            a1[2] = ((_BYTE)v43 + 8) & 0x1F;
          }
          else
          {
            a1[2] = v34;
          }
        }
      }
      else
      {
        v22 = a1[2];
        v23 = (v11 >> 1) & 7;
        v24 = v23 << v22;
        v25 = v22 + 3;
        v26 = a1[3] | v24;
        a1[3] = v26;
        if ( v25 > 0x1F )
        {
          v36 = *(_QWORD *)a1;
          v37 = *(unsigned int *)(*(_QWORD *)a1 + 8LL);
          if ( (unsigned __int64)*(unsigned int *)(*(_QWORD *)a1 + 12LL) - v37 <= 3 )
          {
            v49 = v26;
            v52 = v6;
            sub_16CD150(v36, v36 + 16, v37 + 4, 1);
            v26 = v49;
            v6 = v52;
            v37 = *(unsigned int *)(v36 + 8);
          }
          *(_DWORD *)(*(_QWORD *)v36 + v37) = v26;
          v38 = 0;
          *(_DWORD *)(v36 + 8) += 4;
          v39 = a1[2];
          v40 = v23 >> (32 - v39);
          if ( v39 )
            v38 = v40;
          a1[3] = v38;
          a1[2] = ((_BYTE)v39 + 3) & 0x1F;
        }
        else
        {
          a1[2] = v25;
        }
        v27 = (*(_BYTE *)(v6 + 8) >> 1) & 7;
        if ( v27 > 2u )
        {
          if ( ((v27 + 5) & 7u) > 2 )
LABEL_20:
            sub_16BD130("Invalid encoding", 1);
        }
        else
        {
          if ( !v27 )
            goto LABEL_20;
          sub_1525280(a1, *(_QWORD *)v6, 5);
        }
      }
      v4 += 16;
    }
    while ( v46 != v4 );
  }
}
