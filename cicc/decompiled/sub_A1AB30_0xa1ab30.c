// Function: sub_A1AB30
// Address: 0xa1ab30
//
__int64 __fastcall sub_A1AB30(__int64 a1, __int64 *a2)
{
  __int64 v3; // r14
  __int64 v4; // r13
  __int64 v5; // rax
  int v6; // ecx
  __int64 v7; // r8
  unsigned int v8; // eax
  unsigned int v9; // r9d
  unsigned int v10; // ecx
  int v11; // r9d
  unsigned __int8 v12; // al
  unsigned __int64 v13; // r12
  unsigned int v14; // r8d
  unsigned int v15; // eax
  unsigned int v16; // ecx
  int v17; // edx
  unsigned int v18; // r8d
  unsigned int v19; // r8d
  int v20; // edx
  _QWORD *v21; // rdi
  __int64 v22; // rdx
  int v23; // ecx
  unsigned int v24; // eax
  unsigned int v25; // r9d
  unsigned int v26; // ecx
  int v27; // r9d
  unsigned __int8 v28; // al
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rsi
  _QWORD *v34; // r12
  __int64 v35; // rdx
  int v36; // edx
  unsigned int v37; // eax
  unsigned int v38; // ecx
  int v39; // edx
  unsigned int v40; // ecx
  int v41; // eax
  _QWORD *v42; // rdi
  __int64 v43; // rdx
  int v44; // edx
  unsigned int v45; // eax
  unsigned int v46; // ecx
  _QWORD *v47; // rdi
  __int64 v48; // rdx
  unsigned int v49; // edx
  int v50; // eax
  unsigned int v51; // r8d
  __int64 v53; // [rsp+10h] [rbp-50h]
  int v54; // [rsp+18h] [rbp-48h]
  unsigned int v55; // [rsp+1Ch] [rbp-44h]
  int v56; // [rsp+1Ch] [rbp-44h]
  unsigned int v57; // [rsp+1Ch] [rbp-44h]
  unsigned int v58; // [rsp+1Ch] [rbp-44h]
  unsigned int v59; // [rsp+20h] [rbp-40h]
  __int64 v60; // [rsp+20h] [rbp-40h]
  int v61; // [rsp+20h] [rbp-40h]
  __int64 v62; // [rsp+28h] [rbp-38h]

  v3 = 0;
  v4 = *a2;
  sub_A17B10(a1, 2u, *(_DWORD *)(a1 + 56));
  sub_A17CC0(a1, *(_DWORD *)(v4 + 8), 5);
  v5 = *(unsigned int *)(v4 + 8);
  v53 = 16 * v5;
  if ( (_DWORD)v5 )
  {
    do
    {
      v6 = *(_DWORD *)(a1 + 48);
      v7 = v3 + *(_QWORD *)v4;
      v8 = *(_BYTE *)(v7 + 8) & 1;
      v9 = v8 << v6;
      v10 = v6 + 1;
      v11 = *(_DWORD *)(a1 + 52) | v9;
      *(_DWORD *)(a1 + 52) = v11;
      if ( v10 > 0x1F )
      {
        v34 = *(_QWORD **)(a1 + 24);
        v35 = v34[1];
        if ( (unsigned __int64)(v35 + 4) > v34[2] )
        {
          v56 = v11;
          v59 = v8;
          v62 = v7;
          sub_C8D290(*(_QWORD *)(a1 + 24), v34 + 3, v35 + 4, 1);
          v35 = v34[1];
          v11 = v56;
          v8 = v59;
          v7 = v62;
        }
        *(_DWORD *)(*v34 + v35) = v11;
        v34[1] += 4LL;
        v36 = *(_DWORD *)(a1 + 48);
        v37 = v8 >> (32 - v36);
        v38 = 0;
        if ( v36 )
          v38 = v37;
        *(_DWORD *)(a1 + 52) = v38;
        *(_DWORD *)(a1 + 48) = ((_BYTE)v36 + 1) & 0x1F;
      }
      else
      {
        *(_DWORD *)(a1 + 48) = v10;
      }
      v12 = *(_BYTE *)(v7 + 8);
      if ( (v12 & 1) != 0 )
      {
        v13 = *(_QWORD *)v7;
        v14 = *(_QWORD *)v7;
        if ( v13 == v14 )
        {
          sub_A17CC0(a1, v13, 8);
        }
        else
        {
          v15 = *(_DWORD *)(a1 + 52);
          v16 = *(_DWORD *)(a1 + 48);
          if ( v13 > 0x7F )
          {
            do
            {
              v19 = v13 & 0x7F | 0x80;
              v20 = v19 << v16;
              v16 += 8;
              v15 |= v20;
              *(_DWORD *)(a1 + 52) = v15;
              if ( v16 > 0x1F )
              {
                v21 = *(_QWORD **)(a1 + 24);
                v22 = v21[1];
                if ( (unsigned __int64)(v22 + 4) > v21[2] )
                {
                  v55 = v15;
                  sub_C8D290(v21, v21 + 3, v22 + 4, 1);
                  v15 = v55;
                  v19 = v13 & 0x7F | 0x80;
                  v22 = v21[1];
                }
                *(_DWORD *)(*v21 + v22) = v15;
                v15 = 0;
                v21[1] += 4LL;
                v17 = *(_DWORD *)(a1 + 48);
                v18 = v19 >> (32 - v17);
                if ( v17 )
                  v15 = v18;
                v16 = ((_BYTE)v17 + 8) & 0x1F;
                *(_DWORD *)(a1 + 52) = v15;
              }
              v13 >>= 7;
              *(_DWORD *)(a1 + 48) = v16;
            }
            while ( v13 > 0x7F );
            v14 = v13;
          }
          v39 = v14 << v16;
          v40 = v16 + 8;
          v41 = v39 | v15;
          *(_DWORD *)(a1 + 52) = v41;
          if ( v40 > 0x1F )
          {
            v47 = *(_QWORD **)(a1 + 24);
            v48 = v47[1];
            if ( (unsigned __int64)(v48 + 4) > v47[2] )
            {
              v58 = v14;
              v61 = v41;
              sub_C8D290(v47, v47 + 3, v48 + 4, 1);
              v14 = v58;
              v41 = v61;
              v48 = v47[1];
            }
            *(_DWORD *)(*v47 + v48) = v41;
            v49 = 0;
            v47[1] += 4LL;
            v50 = *(_DWORD *)(a1 + 48);
            v51 = v14 >> (32 - v50);
            if ( v50 )
              v49 = v51;
            *(_DWORD *)(a1 + 52) = v49;
            *(_DWORD *)(a1 + 48) = ((_BYTE)v50 + 8) & 0x1F;
          }
          else
          {
            *(_DWORD *)(a1 + 48) = v40;
          }
        }
      }
      else
      {
        v23 = *(_DWORD *)(a1 + 48);
        v24 = (v12 >> 1) & 7;
        v25 = v24 << v23;
        v26 = v23 + 3;
        v27 = *(_DWORD *)(a1 + 52) | v25;
        *(_DWORD *)(a1 + 52) = v27;
        if ( v26 > 0x1F )
        {
          v42 = *(_QWORD **)(a1 + 24);
          v43 = v42[1];
          if ( (unsigned __int64)(v43 + 4) > v42[2] )
          {
            v54 = v27;
            v57 = v24;
            v60 = v7;
            sub_C8D290(v42, v42 + 3, v43 + 4, 1);
            v27 = v54;
            v24 = v57;
            v7 = v60;
            v43 = v42[1];
          }
          *(_DWORD *)(*v42 + v43) = v27;
          v42[1] += 4LL;
          v44 = *(_DWORD *)(a1 + 48);
          v45 = v24 >> (32 - v44);
          v46 = 0;
          if ( v44 )
            v46 = v45;
          *(_DWORD *)(a1 + 52) = v46;
          *(_DWORD *)(a1 + 48) = ((_BYTE)v44 + 3) & 0x1F;
        }
        else
        {
          *(_DWORD *)(a1 + 48) = v26;
        }
        v28 = (*(_BYTE *)(v7 + 8) >> 1) & 7;
        if ( v28 > 2u )
        {
          if ( ((v28 + 5) & 7u) > 2 )
LABEL_19:
            sub_C64ED0("Invalid encoding", 1);
        }
        else
        {
          if ( !v28 )
            goto LABEL_19;
          sub_A17DE0(a1, *(_QWORD *)v7, 5);
        }
      }
      v3 += 16;
    }
    while ( v53 != v3 );
  }
  v29 = *(_QWORD *)(a1 + 72);
  if ( v29 == *(_QWORD *)(a1 + 80) )
  {
    sub_A1A390((char **)(a1 + 64), (char *)v29, a2);
    v32 = *(_QWORD *)(a1 + 72);
  }
  else
  {
    if ( v29 )
    {
      v30 = *a2;
      *(_QWORD *)(v29 + 8) = 0;
      *a2 = 0;
      *(_QWORD *)v29 = v30;
      v31 = a2[1];
      a2[1] = 0;
      *(_QWORD *)(v29 + 8) = v31;
      v29 = *(_QWORD *)(a1 + 72);
    }
    v32 = v29 + 16;
    *(_QWORD *)(a1 + 72) = v32;
  }
  return (unsigned int)((v32 - *(_QWORD *)(a1 + 64)) >> 4) + 3;
}
