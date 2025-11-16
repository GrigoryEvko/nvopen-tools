// Function: sub_1EBCB20
// Address: 0x1ebcb20
//
__int64 __fastcall sub_1EBCB20(__int64 a1, __int64 *a2, _QWORD *a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v7; // rax
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  int v12; // r12d
  unsigned int *v15; // rbx
  __int64 v16; // rsi
  __int64 *v17; // rax
  __int64 v18; // r15
  __int64 v19; // rdx
  _BOOL4 v20; // r15d
  _QWORD *v21; // rdi
  __int64 *v22; // rdx
  __int64 v23; // rax
  unsigned int v24; // edx
  unsigned int v25; // eax
  __int64 v26; // rdi
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // rcx
  __int64 v31; // rax
  unsigned __int64 v33; // [rsp+10h] [rbp-60h]
  int v34; // [rsp+18h] [rbp-58h]
  unsigned __int64 v35; // [rsp+18h] [rbp-58h]
  __int64 v36; // [rsp+20h] [rbp-50h]
  unsigned __int64 v37; // [rsp+28h] [rbp-48h]
  _QWORD v38[7]; // [rsp+38h] [rbp-38h] BYREF

  v6 = a1;
  v7 = *(_QWORD *)(a1 + 984);
  v36 = *(_QWORD *)(v7 + 280);
  v8 = *(unsigned int *)(v7 + 288);
  v37 = v8;
  v9 = *(unsigned int *)(a1 + 24096);
  if ( v8 >= v9 )
  {
    if ( v8 > v9 )
    {
      if ( v8 > *(unsigned int *)(a1 + 24100) )
      {
        sub_16CD150(a1 + 24088, (const void *)(a1 + 24104), v8, 8, a5, a6);
        v9 = *(unsigned int *)(a1 + 24096);
      }
      v10 = *(_QWORD *)(a1 + 24088);
      v28 = v10 + 8 * v9;
      v29 = v10 + 8 * v8;
      if ( v28 != v29 )
      {
        do
        {
          if ( v28 )
          {
            *(_DWORD *)v28 = 0;
            *(_WORD *)(v28 + 4) = 0;
            *(_BYTE *)(v28 + 6) = 0;
          }
          v28 += 8;
        }
        while ( v29 != v28 );
        v10 = *(_QWORD *)(a1 + 24088);
      }
      *(_DWORD *)(a1 + 24096) = v8;
      v38[0] = 0;
      goto LABEL_5;
    }
  }
  else
  {
    *(_DWORD *)(a1 + 24096) = v8;
  }
  v38[0] = 0;
  if ( v8 )
  {
    v10 = *(_QWORD *)(a1 + 24088);
LABEL_5:
    v11 = 0;
    v12 = 0;
    while ( 1 )
    {
      a4 = v36 + 40 * v11;
      v15 = (unsigned int *)(v10 + 8 * v11);
      v16 = *(unsigned int *)(*(_QWORD *)a4 + 48LL);
      v17 = &qword_4FCF930;
      *v15 = v16;
      v18 = *a2;
      if ( *a2 )
      {
        v17 = (__int64 *)(*(_QWORD *)(v18 + 512) + 24LL * (unsigned int)v16);
        if ( *(_DWORD *)v17 != *(_DWORD *)(v18 + 4) )
        {
          v33 = a4;
          sub_20F85B0(v18, v16, 24LL * (unsigned int)v16, a4, a5, a6);
          a4 = v33;
          v17 = (__int64 *)(24LL * (unsigned int)v16 + *(_QWORD *)(v18 + 512));
        }
      }
      a2[1] = (__int64)v17;
      *((_BYTE *)v15 + 4) = *(_BYTE *)(a4 + 32);
      *((_BYTE *)v15 + 5) = *(_BYTE *)(a4 + 33);
      *((_BYTE *)v15 + 6) = (*(_QWORD *)(a4 + 24) & 0xFFFFFFFFFFFFFFF8LL) != 0;
      if ( (*(_QWORD *)(a2[1] + 8) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_6;
      if ( !*(_BYTE *)(a4 + 32) )
        break;
      v19 = *v15;
      v25 = *(_DWORD *)((*(_QWORD *)(a2[1] + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*(__int64 *)(a2[1] + 8) >> 1) & 3;
      v26 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 800) + 392LL) + 16 * v19);
      if ( v25 > (*(_DWORD *)((v26 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v26 >> 1) & 3) )
      {
        if ( v25 >= (*(_DWORD *)((*(_QWORD *)(a4 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                   | (unsigned int)(*(__int64 *)(a4 + 8) >> 1) & 3) )
        {
          v20 = v25 < (*(_DWORD *)((*(_QWORD *)(a4 + 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                     | (unsigned int)(*(__int64 *)(a4 + 16) >> 1) & 3);
          if ( !*(_BYTE *)(a4 + 33) )
          {
LABEL_30:
            if ( v20 )
              sub_16AF570(v38, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 848) + 376LL) + 8LL * *v15));
            goto LABEL_6;
          }
LABEL_15:
          v21 = *(_QWORD **)(a1 + 984);
          v22 = (__int64 *)(v21[7] + 16LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(*v21 + 96LL) + 8 * v19) + 48LL));
          v23 = *v22;
          if ( (*v22 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v22[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            v35 = a4;
            v23 = sub_1F13A50(v21 + 6, v21[5]);
            a4 = v35;
          }
          v24 = *(_DWORD *)((*(_QWORD *)(a2[1] + 16) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*(__int64 *)(a2[1] + 16) >> 1) & 3;
          if ( v24 < (*(_DWORD *)((v23 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v23 >> 1) & 3) )
          {
            if ( v24 <= (*(_DWORD *)((*(_QWORD *)(a4 + 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                       | (unsigned int)(*(__int64 *)(a4 + 16) >> 1) & 3) )
            {
              v30 = *(_QWORD *)(a4 + 8);
              v31 = v30 >> 1;
              a4 = v30 & 0xFFFFFFFFFFFFFFF8LL;
              if ( v24 <= (*(_DWORD *)(a4 + 24) | (unsigned int)(v31 & 3)) )
                goto LABEL_30;
              v34 = v12;
            }
            else
            {
              v34 = v12;
              *((_BYTE *)v15 + 5) = 2;
            }
          }
          else
          {
            *((_BYTE *)v15 + 5) = 4;
            v34 = v12;
          }
          while ( 1 )
          {
            sub_16AF570(v38, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 848) + 376LL) + 8LL * *v15));
            if ( !v20 )
              break;
            v20 = 0;
          }
          v12 = v34;
LABEL_6:
          v11 = (unsigned int)++v12;
          if ( v12 == v37 )
            goto LABEL_25;
          goto LABEL_7;
        }
        *((_BYTE *)v15 + 4) = 2;
        if ( *(_BYTE *)(a4 + 33) )
        {
LABEL_44:
          v20 = 1;
          goto LABEL_15;
        }
      }
      else
      {
        *((_BYTE *)v15 + 4) = 4;
        if ( *(_BYTE *)(a4 + 33) )
          goto LABEL_44;
      }
      sub_16AF570(v38, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 848) + 376LL) + 8 * v19));
      v11 = (unsigned int)++v12;
      if ( v12 == v37 )
      {
LABEL_25:
        v6 = a1;
        goto LABEL_26;
      }
LABEL_7:
      v10 = *(_QWORD *)(a1 + 24088);
    }
    if ( !*(_BYTE *)(a4 + 33) )
      goto LABEL_6;
    v19 = *v15;
    v20 = 0;
    goto LABEL_15;
  }
LABEL_26:
  *a3 = v38[0];
  sub_1F12700(*(_QWORD *)(v6 + 848), *(_QWORD *)(v6 + 24088), *(unsigned int *)(v6 + 24096), a4, a5, a6);
  return sub_1F12DF0(*(_QWORD *)(v6 + 848));
}
