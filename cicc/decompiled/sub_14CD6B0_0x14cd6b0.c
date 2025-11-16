// Function: sub_14CD6B0
// Address: 0x14cd6b0
//
void __fastcall sub_14CD6B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  _QWORD *v5; // r12
  bool v6; // si
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // r14
  __int64 v10; // rbx
  __int64 v11; // r14
  _QWORD *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // r13
  __int64 v15; // rdx
  __int64 v16; // rdx
  _QWORD *v17; // rcx
  __int64 v18; // rax
  unsigned int v19; // edx
  __int64 v20; // rcx
  __int64 v21; // rdx
  int v22; // r9d
  _QWORD v23[2]; // [rsp+8h] [rbp-48h] BYREF
  __int64 v24; // [rsp+18h] [rbp-38h]
  __int64 v25; // [rsp+20h] [rbp-30h]

  v4 = a2;
  v24 = a2;
  v5 = sub_14CCFE0(a1, a3);
  v23[0] = 2;
  v23[1] = 0;
  v6 = a2 != -16 && a2 != -8 && a2 != 0;
  if ( v6 )
  {
    sub_164C220(v23);
    v4 = v24;
    v6 = v24 != -16 && v24 != -8 && v24 != 0;
  }
  v7 = *(unsigned int *)(a1 + 176);
  v25 = 0;
  v8 = *(_QWORD *)(a1 + 160);
  if ( (_DWORD)v7 )
  {
    v19 = (v7 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v9 = v8 + 88LL * v19;
    v20 = *(_QWORD *)(v9 + 24);
    if ( v20 == v4 )
    {
LABEL_33:
      if ( !v6 )
        goto LABEL_6;
      goto LABEL_5;
    }
    v22 = 1;
    while ( v20 != -8 )
    {
      v19 = (v7 - 1) & (v22 + v19);
      v9 = v8 + 88LL * v19;
      v20 = *(_QWORD *)(v9 + 24);
      if ( v20 == v4 )
        goto LABEL_33;
      ++v22;
    }
  }
  v9 = v8 + 88LL * (unsigned int)v7;
  if ( !v6 )
    return;
LABEL_5:
  sub_1649B30(v23);
  v8 = *(_QWORD *)(a1 + 160);
  v7 = *(unsigned int *)(a1 + 176);
LABEL_6:
  if ( v9 != v8 + 88 * v7 )
  {
    v10 = *(_QWORD *)(v9 + 40);
    v11 = v10 + 32LL * *(unsigned int *)(v9 + 48);
    if ( v10 != v11 )
    {
      while ( 1 )
      {
        v12 = (_QWORD *)*v5;
        v13 = *((unsigned int *)v5 + 2);
        v14 = *v5 + 32 * v13;
        v15 = (32 * v13) >> 5;
        if ( (32 * v13) >> 7 )
        {
          v16 = *(_QWORD *)(v10 + 16);
          v17 = &v12[16 * ((32 * v13) >> 7)];
          while ( v12[2] != v16 )
          {
            if ( v16 == v12[6] )
            {
              v12 += 4;
              goto LABEL_15;
            }
            if ( v16 == v12[10] )
            {
              v12 += 8;
              goto LABEL_15;
            }
            if ( v16 == v12[14] )
            {
              v12 += 12;
              goto LABEL_15;
            }
            v12 += 16;
            if ( v17 == v12 )
            {
              v15 = (v14 - (__int64)v12) >> 5;
              goto LABEL_19;
            }
          }
          goto LABEL_15;
        }
LABEL_19:
        if ( v15 == 2 )
          break;
        if ( v15 == 3 )
        {
          v21 = *(_QWORD *)(v10 + 16);
          if ( v12[2] == v21 )
            goto LABEL_15;
          v12 += 4;
          goto LABEL_41;
        }
        if ( v15 != 1 )
        {
LABEL_22:
          if ( (unsigned int)v13 >= *((_DWORD *)v5 + 3) )
            goto LABEL_38;
          goto LABEL_23;
        }
        v21 = *(_QWORD *)(v10 + 16);
LABEL_36:
        if ( v12[2] != v21 )
        {
          if ( (unsigned int)v13 >= *((_DWORD *)v5 + 3) )
          {
LABEL_38:
            sub_14CB640((__int64)v5, 0);
            v13 = *((unsigned int *)v5 + 2);
            v14 = *v5 + 32 * v13;
          }
LABEL_23:
          if ( v14 )
          {
            *(_QWORD *)v14 = 6;
            *(_QWORD *)(v14 + 8) = 0;
            v18 = *(_QWORD *)(v10 + 16);
            *(_QWORD *)(v14 + 16) = v18;
            if ( v18 != 0 && v18 != -8 && v18 != -16 )
              sub_1649AC0(v14, *(_QWORD *)v10 & 0xFFFFFFFFFFFFFFF8LL);
            *(_DWORD *)(v14 + 24) = *(_DWORD *)(v10 + 24);
            LODWORD(v13) = *((_DWORD *)v5 + 2);
          }
          *((_DWORD *)v5 + 2) = v13 + 1;
          goto LABEL_16;
        }
LABEL_15:
        if ( (_QWORD *)v14 == v12 )
          goto LABEL_22;
LABEL_16:
        v10 += 32;
        if ( v11 == v10 )
          return;
      }
      v21 = *(_QWORD *)(v10 + 16);
LABEL_41:
      if ( v12[2] == v21 )
        goto LABEL_15;
      v12 += 4;
      goto LABEL_36;
    }
  }
}
