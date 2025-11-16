// Function: sub_3509FC0
// Address: 0x3509fc0
//
__int64 __fastcall sub_3509FC0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  signed __int64 v4; // rbx
  __int64 v5; // r8
  __int64 v6; // r12
  __int64 v7; // r14
  signed __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r9
  __int64 v11; // rdi
  __int64 (*v12)(); // rax
  __int64 v14; // r10
  unsigned int v15; // eax
  unsigned __int64 v16; // rcx
  __int64 v17; // r15
  __int64 v18; // r13
  __int64 *v19; // rdx
  __int64 v20; // r15
  __int64 *v21; // rdx
  __int64 v22; // rdi
  __int64 v23; // rdx
  unsigned __int16 v24; // ax
  __int64 *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r15
  __int64 v29; // r13
  __int64 v30; // rbx
  __int64 v31; // r14
  __int64 *v32; // rax
  unsigned int v33; // eax
  __int64 v34; // rsi
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  unsigned __int64 v38; // rdx
  __int64 v39; // rax
  unsigned __int64 v40; // r13
  _QWORD *v41; // rdx
  _QWORD *v42; // rdi
  __int64 v43; // [rsp+0h] [rbp-70h]
  int v44; // [rsp+Ch] [rbp-64h]
  unsigned __int64 v45; // [rsp+10h] [rbp-60h]
  __int64 v46; // [rsp+18h] [rbp-58h]
  _QWORD *v47; // [rsp+18h] [rbp-58h]
  __int64 v48; // [rsp+18h] [rbp-58h]
  unsigned __int64 v49; // [rsp+20h] [rbp-50h]
  unsigned __int64 v51; // [rsp+30h] [rbp-40h]
  unsigned int v52; // [rsp+38h] [rbp-38h]
  unsigned __int8 v53; // [rsp+3Fh] [rbp-31h]

  v4 = a4;
  v49 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  v45 = a3 & 0xFFFFFFFFFFFFFFF8LL | 2;
  v5 = *(_QWORD *)(a2 + 32);
  if ( (*(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a4 >> 1) & 3) < (*(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL)
                                                                                                  + 24)
                                                                                      | 1u) )
    v4 = a4 & 0xFFFFFFFFFFFFFFF8LL | 2;
  v6 = v5 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
  if ( v6 != v5 )
  {
    v7 = v4;
    v52 = (v4 >> 1) & 3;
    v8 = v4;
    v9 = *(_QWORD *)(a2 + 32);
    v51 = v8 & 0xFFFFFFFFFFFFFFF8LL;
    while ( 1 )
    {
      while ( 1 )
      {
        if ( *(_BYTE *)v9 )
          goto LABEL_36;
        v10 = *(unsigned int *)(v9 + 8);
        if ( !(_DWORD)v10 )
          goto LABEL_36;
        v53 = *(_BYTE *)(v9 + 4) & 1;
        if ( v53 || (*(_BYTE *)(v9 + 4) & 2) != 0 || (*(_BYTE *)(v9 + 3) & 0x10) != 0 && (*(_DWORD *)v9 & 0xFFF00) == 0 )
          goto LABEL_36;
        if ( (unsigned int)(v10 - 1) > 0x3FFFFFFE )
          break;
        if ( (unsigned __int8)sub_2EBF3A0((_QWORD *)a1[3], v10) )
          goto LABEL_36;
        v11 = a1[6];
        v12 = *(__int64 (**)())(*(_QWORD *)v11 + 32LL);
        if ( v12 == sub_2E4EE60 || !((unsigned __int8 (__fastcall *)(__int64, __int64))v12)(v11, v9) )
          return v53;
        v9 += 40;
        if ( v6 == v9 )
          return 1;
      }
      v14 = a1[4];
      v15 = v10 & 0x7FFFFFFF;
      v16 = *(unsigned int *)(v14 + 160);
      v17 = 8 * (v10 & 0x7FFFFFFF);
      if ( ((unsigned int)v10 & 0x7FFFFFFF) >= (unsigned int)v16 )
        break;
      v18 = *(_QWORD *)(*(_QWORD *)(v14 + 152) + 8LL * v15);
      if ( !v18 )
        break;
LABEL_17:
      v19 = (__int64 *)sub_2E09D00((__int64 *)v18, v45);
      if ( v19 != (__int64 *)(*(_QWORD *)v18 + 24LL * *(unsigned int *)(v18 + 8))
        && (*(_DWORD *)((*v19 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v19 >> 1) & 3) <= (*(_DWORD *)(v49 + 24)
                                                                                               | 1u) )
      {
        v20 = v19[2];
        if ( v20 )
        {
          if ( v51 == v49 )
            return v53;
          v21 = (__int64 *)sub_2E09D00((__int64 *)v18, v7);
          if ( v21 == (__int64 *)(*(_QWORD *)v18 + 24LL * *(unsigned int *)(v18 + 8))
            || (*(_DWORD *)((*v21 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v21 >> 1) & 3) > (*(_DWORD *)(v51 + 24)
                                                                                                  | v52)
            || v20 != v21[2] )
          {
            return v53;
          }
          if ( *(_QWORD *)(v18 + 104) )
          {
            v22 = *(_QWORD *)(*(_QWORD *)a1[3] + 16LL);
            v23 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v22 + 200LL))(v22);
            v24 = (*(_DWORD *)v9 >> 8) & 0xFFF;
            if ( v24 )
            {
              v25 = (__int64 *)(*(_QWORD *)(v23 + 272) + 16LL * v24);
              v26 = *v25;
              v27 = v25[1];
            }
            else
            {
              v36 = sub_2EBF1E0(a1[3], *(_DWORD *)(v9 + 8));
              v27 = v37;
              v26 = v36;
            }
            v28 = *(_QWORD *)(v18 + 104);
            if ( v28 )
            {
              v46 = v9;
              v29 = v26;
              v30 = v7;
              v31 = v27;
              do
              {
                if ( v31 & *(_QWORD *)(v28 + 120) | v29 & *(_QWORD *)(v28 + 112) )
                {
                  v32 = (__int64 *)sub_2E09D00((__int64 *)v28, v30);
                  if ( v32 == (__int64 *)(*(_QWORD *)v28 + 24LL * *(unsigned int *)(v28 + 8))
                    || (*(_DWORD *)((*v32 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v32 >> 1) & 3) > (*(_DWORD *)(v51 + 24) | v52) )
                  {
                    return v53;
                  }
                  v31 &= ~*(_QWORD *)(v28 + 120);
                  v29 &= ~*(_QWORD *)(v28 + 112);
                  if ( !v29 && !v31 )
                    break;
                }
                v28 = *(_QWORD *)(v28 + 104);
              }
              while ( v28 );
              v7 = v30;
              v9 = v46;
            }
          }
        }
      }
LABEL_36:
      v9 += 40;
      if ( v6 == v9 )
        return 1;
    }
    v33 = v15 + 1;
    if ( (unsigned int)v16 < v33 )
    {
      v38 = v33;
      if ( v33 != v16 )
      {
        if ( v33 >= v16 )
        {
          v39 = *(_QWORD *)(v14 + 168);
          v40 = v38 - v16;
          if ( v38 > *(unsigned int *)(v14 + 164) )
          {
            v43 = *(_QWORD *)(v14 + 168);
            v44 = *(_DWORD *)(v9 + 8);
            v48 = a1[4];
            sub_C8D5F0(v14 + 152, (const void *)(v14 + 168), v38, 8u, v5, v10);
            v14 = v48;
            v39 = v43;
            LODWORD(v10) = v44;
            v16 = *(unsigned int *)(v48 + 160);
          }
          v34 = *(_QWORD *)(v14 + 152);
          v41 = (_QWORD *)(v34 + 8 * v16);
          v42 = &v41[v40];
          if ( v41 != v42 )
          {
            do
              *v41++ = v39;
            while ( v42 != v41 );
            LODWORD(v16) = *(_DWORD *)(v14 + 160);
            v34 = *(_QWORD *)(v14 + 152);
          }
          *(_DWORD *)(v14 + 160) = v40 + v16;
          goto LABEL_40;
        }
        *(_DWORD *)(v14 + 160) = v33;
      }
    }
    v34 = *(_QWORD *)(v14 + 152);
LABEL_40:
    v47 = (_QWORD *)v14;
    v35 = sub_2E10F30(v10);
    *(_QWORD *)(v34 + v17) = v35;
    v18 = v35;
    sub_2E11E80(v47, v35);
    goto LABEL_17;
  }
  return 1;
}
