// Function: sub_264D7D0
// Address: 0x264d7d0
//
unsigned __int64 *__fastcall sub_264D7D0(__int64 a1, __int64 a2)
{
  unsigned __int64 *result; // rax
  unsigned __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r15
  _DWORD *v13; // r12
  __int64 v14; // r9
  unsigned int *v15; // r15
  unsigned int *v16; // r12
  unsigned __int64 v17; // rax
  char *v18; // r15
  unsigned int v19; // r12d
  __int64 v20; // rax
  __int64 v21; // r12
  __int64 v22; // r15
  __int64 v23; // r12
  __int64 v24; // r15
  char v25; // r12
  unsigned __int64 *v26; // rax
  unsigned __int64 *v27; // r13
  __int64 v28; // rdi
  unsigned __int64 v29; // r15
  _WORD *v30; // rdx
  _BYTE *v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // r12
  __int64 v35; // rax
  __int64 v36; // rax
  unsigned __int64 *v37; // [rsp+10h] [rbp-F0h]
  unsigned __int64 *v38; // [rsp+20h] [rbp-E0h]
  char *v39; // [rsp+28h] [rbp-D8h]
  __int64 v40; // [rsp+28h] [rbp-D8h]
  __int64 v41; // [rsp+28h] [rbp-D8h]
  unsigned __int64 *v42; // [rsp+28h] [rbp-D8h]
  __int64 v43; // [rsp+28h] [rbp-D8h]
  char *v44; // [rsp+30h] [rbp-D0h] BYREF
  char *v45; // [rsp+38h] [rbp-C8h]
  __int64 v46; // [rsp+40h] [rbp-C0h]
  __int64 v47[3]; // [rsp+50h] [rbp-B0h] BYREF
  unsigned int v48; // [rsp+68h] [rbp-98h]
  __int64 *v49; // [rsp+70h] [rbp-90h]
  __int64 v50; // [rsp+78h] [rbp-88h]
  _DWORD *v51; // [rsp+80h] [rbp-80h]
  _DWORD *v52; // [rsp+88h] [rbp-78h]
  __m128i v53; // [rsp+90h] [rbp-70h] BYREF
  _DWORD *v54; // [rsp+A0h] [rbp-60h]
  _DWORD *v55; // [rsp+A8h] [rbp-58h]
  __m128i v56; // [rsp+B0h] [rbp-50h] BYREF
  _DWORD *v57; // [rsp+C0h] [rbp-40h]
  _DWORD *v58; // [rsp+C8h] [rbp-38h]

  sub_904010(a2, "Callsite Context Graph:\n");
  result = *(unsigned __int64 **)(a1 + 328);
  v37 = result;
  v38 = *(unsigned __int64 **)(a1 + 320);
  if ( result != v38 )
  {
    do
    {
      v3 = *v38;
      if ( *(_BYTE *)(*v38 + 2) )
      {
        v4 = sub_904010(a2, "Node ");
        v5 = sub_CB5A80(v4, v3);
        sub_904010(v5, "\n");
        sub_904010(a2, "\t");
        v6 = *(_QWORD *)(v3 + 8);
        if ( v6 )
        {
          sub_A69870(v6, (_BYTE *)a2, 0);
          v7 = sub_904010(a2, "\t(clone ");
          v8 = sub_CB59D0(v7, *(unsigned int *)(v3 + 16));
          sub_904010(v8, ")");
        }
        else
        {
          sub_904010(a2, "null Call");
        }
        if ( *(_BYTE *)(v3 + 1) )
          sub_904010(a2, " (recursive)");
        sub_904010(a2, "\n");
        if ( *(_DWORD *)(v3 + 32) )
        {
          sub_904010(a2, "\tMatchingCalls:\n");
          v34 = *(_QWORD *)(v3 + 24);
          v43 = v34 + 16LL * *(unsigned int *)(v3 + 32);
          while ( v43 != v34 )
          {
            sub_904010(a2, "\t");
            if ( *(_QWORD *)v34 )
            {
              sub_A69870(*(_QWORD *)v34, (_BYTE *)a2, 0);
              v35 = sub_904010(a2, "\t(clone ");
              v36 = sub_CB59D0(v35, *(unsigned int *)(v34 + 8));
              sub_904010(v36, ")");
            }
            else
            {
              sub_904010(a2, "null Call");
            }
            v34 += 16;
            sub_904010(a2, "\n");
          }
        }
        v9 = sub_904010(a2, "\tAllocTypes: ");
        sub_2643CE0(&v56, *(_BYTE *)(v3 + 2));
        v10 = sub_CB6200(v9, (unsigned __int8 *)v56.m128i_i64[0], v56.m128i_u64[1]);
        sub_904010(v10, "\n");
        sub_2240A30((unsigned __int64 *)&v56);
        sub_904010(a2, "\tContextIds:");
        sub_264D230((__int64)v47, v3, v11);
        v12 = v47[0];
        v13 = (_DWORD *)(v47[1] + 4LL * v48);
        sub_22B0690(&v53, v47);
        v50 = v12;
        v51 = v13;
        v52 = v13;
        v49 = v47;
        v58 = v55;
        v56 = v53;
        v57 = v54;
        v44 = 0;
        v45 = 0;
        v46 = 0;
        sub_2640D80(
          (__int64 *)&v44,
          (__int64)v54,
          v53.m128i_i64[1],
          (__int64)v47,
          (__int64)&v44,
          v14,
          v53.m128i_i32[0],
          v53.m128i_i32[2],
          v54,
          v55,
          (int)v47,
          v12,
          v13);
        v15 = (unsigned int *)v45;
        v16 = (unsigned int *)v44;
        if ( v45 != v44 )
        {
          _BitScanReverse64(&v17, (v45 - v44) >> 2);
          sub_263F8F0(v44, v45, 2LL * (int)(63 - (v17 ^ 0x3F)));
          sub_263F470(v16, v15);
          v39 = v45;
          if ( v45 != v44 )
          {
            v18 = v44;
            do
            {
              v19 = *(_DWORD *)v18;
              v18 += 4;
              v20 = sub_904010(a2, " ");
              sub_CB59D0(v20, v19);
            }
            while ( v39 != v18 );
          }
        }
        sub_904010(a2, "\n");
        sub_904010(a2, "\tCalleeEdges:\n");
        v21 = *(_QWORD *)(v3 + 48);
        v40 = *(_QWORD *)(v3 + 56);
        while ( v40 != v21 )
        {
          v21 += 16;
          v22 = sub_904010(a2, "\t\t");
          sub_26444C0(*(_QWORD *)(v21 - 16), v22);
          sub_904010(v22, "\n");
        }
        sub_904010(a2, "\tCallerEdges:\n");
        v23 = *(_QWORD *)(v3 + 72);
        v41 = *(_QWORD *)(v3 + 80);
        while ( v41 != v23 )
        {
          v23 += 16;
          v24 = sub_904010(a2, "\t\t");
          sub_26444C0(*(_QWORD *)(v23 - 16), v24);
          sub_904010(v24, "\n");
        }
        if ( *(_QWORD *)(v3 + 104) == *(_QWORD *)(v3 + 96) )
        {
          if ( *(_QWORD *)(v3 + 120) )
          {
            v32 = sub_904010(a2, "\tClone of ");
            v33 = sub_CB5A80(v32, *(_QWORD *)(v3 + 120));
            sub_904010(v33, "\n");
          }
        }
        else
        {
          v25 = 1;
          sub_904010(a2, "\tClones: ");
          v26 = *(unsigned __int64 **)(v3 + 104);
          v27 = *(unsigned __int64 **)(v3 + 96);
          v42 = v26;
          while ( v42 != v27 )
          {
            v29 = *v27;
            if ( v25 )
            {
              v28 = a2;
              v25 = 0;
            }
            else
            {
              v30 = *(_WORD **)(a2 + 32);
              if ( *(_QWORD *)(a2 + 24) - (_QWORD)v30 > 1u )
              {
                v28 = a2;
                *v30 = 8236;
                *(_QWORD *)(a2 + 32) += 2LL;
              }
              else
              {
                v28 = sub_CB6200(a2, (unsigned __int8 *)", ", 2u);
              }
            }
            ++v27;
            sub_CB5A80(v28, v29);
          }
          sub_904010(a2, "\n");
        }
        if ( v44 )
          j_j___libc_free_0((unsigned __int64)v44);
        sub_2342640((__int64)v47);
        v31 = *(_BYTE **)(a2 + 32);
        if ( *(_BYTE **)(a2 + 24) == v31 )
        {
          sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *v31 = 10;
          ++*(_QWORD *)(a2 + 32);
        }
      }
      result = ++v38;
    }
    while ( v37 != v38 );
  }
  return result;
}
