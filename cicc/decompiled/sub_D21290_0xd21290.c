// Function: sub_D21290
// Address: 0xd21290
//
__int64 __fastcall sub_D21290(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // r15
  __int64 v5; // rbx
  int v6; // ecx
  __int64 v7; // r12
  unsigned int v8; // esi
  int v9; // r11d
  __int64 v10; // r9
  _QWORD *v11; // rdx
  unsigned int v12; // r8d
  _QWORD *v13; // rax
  __int64 v14; // rdi
  _DWORD *v15; // rax
  int v17; // eax
  int v18; // edi
  int v19; // eax
  int v20; // r8d
  __int64 v21; // rsi
  unsigned int v22; // eax
  __int64 v23; // r9
  int v24; // r11d
  _QWORD *v25; // r10
  int v26; // eax
  int v27; // eax
  __int64 v28; // r8
  _QWORD *v29; // r9
  unsigned int v30; // r14d
  int v31; // r10d
  __int64 v32; // rsi
  __int64 v33; // [rsp+8h] [rbp-B8h]
  int v34; // [rsp+1Ch] [rbp-A4h]
  int v35; // [rsp+1Ch] [rbp-A4h]
  int v36; // [rsp+1Ch] [rbp-A4h]
  _QWORD v37[2]; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v38; // [rsp+30h] [rbp-90h]
  __int64 v39; // [rsp+38h] [rbp-88h]
  __int64 v40; // [rsp+40h] [rbp-80h]
  __int64 v41; // [rsp+48h] [rbp-78h]
  __int64 v42; // [rsp+50h] [rbp-70h]
  __int64 v43; // [rsp+58h] [rbp-68h]
  __int64 v44; // [rsp+60h] [rbp-60h]
  __int64 v45; // [rsp+68h] [rbp-58h]
  __int64 v46; // [rsp+70h] [rbp-50h]
  __int64 v47; // [rsp+78h] [rbp-48h]
  __int64 v48; // [rsp+80h] [rbp-40h]
  __int64 v49; // [rsp+88h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 56);
  v37[0] = 0;
  v37[1] = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  sub_D126D0((__int64)v37, v3);
  sub_D12BD0((__int64)v37);
  v4 = v45;
  v5 = v44;
  v6 = 0;
  v33 = a1 + 304;
  if ( v44 != v45 )
  {
    while ( 1 )
    {
      v7 = *(_QWORD *)(*(_QWORD *)v5 + 8LL);
      if ( !v7 )
        goto LABEL_7;
      v8 = *(_DWORD *)(a1 + 328);
      if ( !v8 )
        break;
      v9 = 1;
      v10 = *(_QWORD *)(a1 + 312);
      v11 = 0;
      v12 = (v8 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v13 = (_QWORD *)(v10 + 16LL * v12);
      v14 = *v13;
      if ( v7 != *v13 )
      {
        while ( v14 != -4096 )
        {
          if ( v14 == -8192 && !v11 )
            v11 = v13;
          v12 = (v8 - 1) & (v9 + v12);
          v13 = (_QWORD *)(v10 + 16LL * v12);
          v14 = *v13;
          if ( v7 == *v13 )
            goto LABEL_5;
          ++v9;
        }
        if ( !v11 )
          v11 = v13;
        v17 = *(_DWORD *)(a1 + 320);
        ++*(_QWORD *)(a1 + 304);
        v18 = v17 + 1;
        if ( 4 * (v17 + 1) < 3 * v8 )
        {
          if ( v8 - *(_DWORD *)(a1 + 324) - v18 <= v8 >> 3 )
          {
            v36 = v6;
            sub_D1FCE0(v33, v8);
            v26 = *(_DWORD *)(a1 + 328);
            if ( !v26 )
            {
LABEL_53:
              ++*(_DWORD *)(a1 + 320);
              BUG();
            }
            v27 = v26 - 1;
            v28 = *(_QWORD *)(a1 + 312);
            v29 = 0;
            v30 = v27 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
            v6 = v36;
            v31 = 1;
            v18 = *(_DWORD *)(a1 + 320) + 1;
            v11 = (_QWORD *)(v28 + 16LL * v30);
            v32 = *v11;
            if ( v7 != *v11 )
            {
              while ( v32 != -4096 )
              {
                if ( !v29 && v32 == -8192 )
                  v29 = v11;
                v30 = v27 & (v31 + v30);
                v11 = (_QWORD *)(v28 + 16LL * v30);
                v32 = *v11;
                if ( v7 == *v11 )
                  goto LABEL_26;
                ++v31;
              }
              if ( v29 )
                v11 = v29;
            }
          }
          goto LABEL_26;
        }
LABEL_30:
        v35 = v6;
        sub_D1FCE0(v33, 2 * v8);
        v19 = *(_DWORD *)(a1 + 328);
        if ( !v19 )
          goto LABEL_53;
        v20 = v19 - 1;
        v21 = *(_QWORD *)(a1 + 312);
        v22 = (v19 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v18 = *(_DWORD *)(a1 + 320) + 1;
        v6 = v35;
        v11 = (_QWORD *)(v21 + 16LL * v22);
        v23 = *v11;
        if ( v7 != *v11 )
        {
          v24 = 1;
          v25 = 0;
          while ( v23 != -4096 )
          {
            if ( v23 == -8192 && !v25 )
              v25 = v11;
            v22 = v20 & (v24 + v22);
            v11 = (_QWORD *)(v21 + 16LL * v22);
            v23 = *v11;
            if ( v7 == *v11 )
              goto LABEL_26;
            ++v24;
          }
          if ( v25 )
            v11 = v25;
        }
LABEL_26:
        *(_DWORD *)(a1 + 320) = v18;
        if ( *v11 != -4096 )
          --*(_DWORD *)(a1 + 324);
        *v11 = v7;
        v15 = v11 + 1;
        *((_DWORD *)v11 + 2) = 0;
        goto LABEL_6;
      }
LABEL_5:
      v15 = v13 + 1;
LABEL_6:
      *v15 = v6;
LABEL_7:
      v5 += 8;
      if ( v4 == v5 )
      {
        v34 = v6 + 1;
        sub_D12BD0((__int64)v37);
        v4 = v45;
        v5 = v44;
        v6 = v34;
        if ( v44 == v45 )
          goto LABEL_9;
      }
    }
    ++*(_QWORD *)(a1 + 304);
    goto LABEL_30;
  }
LABEL_9:
  if ( v47 )
    j_j___libc_free_0(v47, v49 - v47);
  if ( v44 )
    j_j___libc_free_0(v44, v46 - v44);
  if ( v41 )
    j_j___libc_free_0(v41, v43 - v41);
  return sub_C7D6A0(v38, 16LL * (unsigned int)v40, 8);
}
