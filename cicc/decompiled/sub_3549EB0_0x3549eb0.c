// Function: sub_3549EB0
// Address: 0x3549eb0
//
__int64 __fastcall sub_3549EB0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r14
  __int64 v3; // r15
  _QWORD *v4; // r13
  _QWORD *v5; // rsi
  _QWORD *v6; // r9
  _QWORD *v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 *v14; // rbx
  __int64 v15; // r12
  _QWORD *v16; // r14
  __int64 v17; // r13
  int v18; // eax
  unsigned __int64 v19; // r15
  _QWORD *v20; // rcx
  _QWORD *v21; // rdi
  _QWORD *v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rdx
  int v25; // eax
  _QWORD *v26; // r8
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rax
  _QWORD *v30; // rax
  _QWORD *v31; // rdx
  char v32; // di
  unsigned __int64 v34; // rdi
  unsigned __int64 v36; // [rsp+8h] [rbp-68h]
  __int64 v37; // [rsp+10h] [rbp-60h]
  __int64 v38; // [rsp+18h] [rbp-58h]
  _QWORD *v39; // [rsp+18h] [rbp-58h]
  int v40; // [rsp+20h] [rbp-50h]
  int v41; // [rsp+24h] [rbp-4Ch]
  __int64 v42; // [rsp+28h] [rbp-48h]
  unsigned __int64 v43; // [rsp+30h] [rbp-40h] BYREF
  unsigned __int64 *v44; // [rsp+38h] [rbp-38h] BYREF

  v42 = *(_QWORD *)(a2 + 56);
  if ( *(_QWORD *)(a2 + 48) != v42 )
  {
    v2 = *(_QWORD *)(a2 + 48);
    v3 = a1;
    v4 = (_QWORD *)(a1 + 40);
    while ( 1 )
    {
LABEL_4:
      if ( (*(_BYTE *)(v2 + 248) & 0x40) == 0 )
        goto LABEL_3;
      v5 = *(_QWORD **)(v3 + 48);
      if ( !v5 )
        break;
      v6 = v4;
      v7 = *(_QWORD **)(v3 + 48);
      do
      {
        while ( 1 )
        {
          v8 = v7[2];
          v9 = v7[3];
          if ( v7[4] >= v2 )
            break;
          v7 = (_QWORD *)v7[3];
          if ( !v9 )
            goto LABEL_10;
        }
        v6 = v7;
        v7 = (_QWORD *)v7[2];
      }
      while ( v8 );
LABEL_10:
      v41 = -1;
      if ( v6 != v4 && v6[4] <= v2 )
        v41 = (*((_DWORD *)v6 + 10) - *(_DWORD *)(v3 + 80)) / *(_DWORD *)(v3 + 88);
      v43 = v2;
      v10 = (__int64)v4;
      do
      {
        while ( 1 )
        {
          v11 = v5[2];
          v12 = v5[3];
          if ( v5[4] >= v2 )
            break;
          v5 = (_QWORD *)v5[3];
          if ( !v12 )
            goto LABEL_17;
        }
        v10 = (__int64)v5;
        v5 = (_QWORD *)v5[2];
      }
      while ( v11 );
LABEL_17:
      if ( v4 == (_QWORD *)v10 || *(_QWORD *)(v10 + 32) > v2 )
        goto LABEL_19;
LABEL_20:
      v40 = *(_DWORD *)(v10 + 40);
      v13 = sub_3545E90(*(_QWORD **)(a2 + 3464), v2);
      v14 = *(__int64 **)v13;
      if ( *(_QWORD *)v13 + 32LL * *(unsigned int *)(v13 + 8) != *(_QWORD *)v13 )
      {
        v36 = v2;
        v15 = *(_QWORD *)v13 + 32LL * *(unsigned int *)(v13 + 8);
        v16 = v4;
        v17 = v3;
        while ( 1 )
        {
          if ( (v14[1] & 6) == 0 )
          {
            v18 = *((_DWORD *)v14 + 4);
            if ( v18 )
            {
              v19 = *v14;
              if ( *(_DWORD *)(*v14 + 200) != -1 && (unsigned int)(v18 - 1) <= 0x3FFFFFFE )
                break;
            }
          }
LABEL_47:
          v14 += 4;
          if ( (__int64 *)v15 == v14 )
          {
            v3 = v17;
            v4 = v16;
            v2 = v36 + 256;
            if ( v42 != v36 + 256 )
              goto LABEL_4;
            return 1;
          }
        }
        v20 = *(_QWORD **)(v17 + 48);
        if ( v20 )
        {
          v21 = v16;
          v22 = *(_QWORD **)(v17 + 48);
          do
          {
            while ( 1 )
            {
              v23 = v22[2];
              v24 = v22[3];
              if ( v22[4] >= v19 )
                break;
              v22 = (_QWORD *)v22[3];
              if ( !v24 )
                goto LABEL_31;
            }
            v21 = v22;
            v22 = (_QWORD *)v22[2];
          }
          while ( v23 );
LABEL_31:
          v25 = -1;
          if ( v21 != v16 && v21[4] <= v19 )
            v25 = (*((_DWORD *)v21 + 10) - *(_DWORD *)(v17 + 80)) / *(_DWORD *)(v17 + 88);
          if ( v25 != v41 )
            return 0;
          v26 = v16;
          do
          {
            while ( 1 )
            {
              v27 = v20[2];
              v28 = v20[3];
              if ( v20[4] >= v19 )
                break;
              v20 = (_QWORD *)v20[3];
              if ( !v28 )
                goto LABEL_39;
            }
            v26 = v20;
            v20 = (_QWORD *)v20[2];
          }
          while ( v27 );
LABEL_39:
          if ( v16 == v26 || v26[4] > v19 )
          {
LABEL_41:
            v37 = (__int64)v26;
            v29 = sub_22077B0(0x30u);
            *(_QWORD *)(v29 + 32) = v19;
            *(_DWORD *)(v29 + 40) = 0;
            v38 = v29;
            v30 = sub_3549D00((_QWORD *)(v17 + 32), v37, (unsigned __int64 *)(v29 + 32));
            if ( !v31 )
            {
              v34 = v38;
              v39 = v30;
              j_j___libc_free_0(v34);
              if ( *((_DWORD *)v39 + 10) <= v40 )
                return 0;
              goto LABEL_47;
            }
            v32 = v16 == v31 || v30 || v19 < v31[4];
            sub_220F040(v32, v38, v31, v16);
            ++*(_QWORD *)(v17 + 72);
            v26 = (_QWORD *)v38;
          }
          if ( *((_DWORD *)v26 + 10) <= v40 )
            return 0;
          goto LABEL_47;
        }
        if ( v41 != -1 )
          return 0;
        v26 = v16;
        goto LABEL_41;
      }
LABEL_3:
      v2 += 256LL;
      if ( v42 == v2 )
        return 1;
    }
    v43 = v2;
    v10 = (__int64)v4;
    v41 = -1;
LABEL_19:
    v44 = &v43;
    v10 = sub_3549E00((_QWORD *)(v3 + 32), v10, &v44);
    goto LABEL_20;
  }
  return 1;
}
