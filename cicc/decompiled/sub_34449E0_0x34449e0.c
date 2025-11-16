// Function: sub_34449E0
// Address: 0x34449e0
//
__int64 __fastcall sub_34449E0(__int64 *a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  unsigned __int16 *v7; // rdx
  int v8; // eax
  __int64 v9; // r15
  unsigned int v10; // r8d
  __int64 v11; // rax
  __int64 v13; // rdx
  bool v14; // zf
  __int64 v15; // rdx
  unsigned int v16; // r9d
  __int64 *v17; // rdx
  unsigned int v18; // r15d
  __int64 v19; // r14
  __int64 v20; // r10
  __int64 v21; // rcx
  unsigned int v22; // esi
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // rdx
  int v27; // eax
  __int64 v28; // rsi
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdx
  int v33; // eax
  __int64 v34; // rdx
  __int64 v35; // r9
  unsigned int v36; // r15d
  __int64 v37; // rax
  __int64 v38; // rax
  int v39; // eax
  int v40; // eax
  __int64 v41; // [rsp+0h] [rbp-C0h]
  __int64 v42; // [rsp+18h] [rbp-A8h]
  unsigned int v43; // [rsp+18h] [rbp-A8h]
  __int64 v44; // [rsp+18h] [rbp-A8h]
  __int64 v45; // [rsp+18h] [rbp-A8h]
  __int64 v46; // [rsp+20h] [rbp-A0h]
  __int64 v47; // [rsp+28h] [rbp-98h]
  __int64 v48; // [rsp+30h] [rbp-90h]
  __int64 v49; // [rsp+30h] [rbp-90h]
  __int64 v50; // [rsp+38h] [rbp-88h]
  unsigned __int8 v51; // [rsp+38h] [rbp-88h]
  __int64 v52; // [rsp+38h] [rbp-88h]
  __int16 v53; // [rsp+60h] [rbp-60h] BYREF
  __int64 v54; // [rsp+68h] [rbp-58h]
  unsigned __int64 v55; // [rsp+70h] [rbp-50h] BYREF
  __int64 v56; // [rsp+78h] [rbp-48h]
  __int64 v57; // [rsp+80h] [rbp-40h]
  __int64 v58; // [rsp+88h] [rbp-38h]

  v7 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * a3);
  v8 = *v7;
  v9 = *((_QWORD *)v7 + 1);
  v53 = v8;
  v54 = v9;
  if ( !(_WORD)v8 )
  {
    if ( !sub_30070B0((__int64)&v53) )
    {
      v56 = v9;
      LOWORD(v55) = 0;
      goto LABEL_10;
    }
    LOWORD(v8) = sub_3009970((__int64)&v53, a2, v23, v24, v25);
LABEL_9:
    LOWORD(v55) = v8;
    v56 = v13;
    if ( (_WORD)v8 )
      goto LABEL_4;
LABEL_10:
    v11 = sub_3007260((__int64)&v55);
    v10 = 0;
    v14 = *(_DWORD *)(a2 + 24) == 187;
    v57 = v11;
    v58 = v15;
    if ( !v14 )
      return v10;
LABEL_11:
    v16 = v11;
    if ( (v11 & 1) != 0 )
      return v10;
    v17 = *(__int64 **)(a2 + 40);
    v18 = (unsigned int)v11 >> 1;
    v19 = *v17;
    v20 = v17[5];
    v50 = *v17;
    v48 = v17[1];
    v47 = v20;
    v21 = v17[6];
    LODWORD(v56) = v11;
    v46 = v21;
    if ( (unsigned int)v11 > 0x40 )
    {
      v42 = v20;
      sub_C43690((__int64)&v55, 0, 0);
      v16 = v56;
      v20 = v42;
    }
    else
    {
      v55 = 0;
    }
    v22 = v16 - v18;
    if ( v16 - v18 != v16 )
    {
      if ( v22 > 0x3F || v16 > 0x40 )
      {
        v45 = v20;
        sub_C43C90(&v55, v22, v16);
        v20 = v45;
      }
      else
      {
        v55 |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v18) << v22;
      }
    }
    if ( *(_DWORD *)(v20 + 24) == 190 )
    {
      v26 = *(_QWORD *)(*(_QWORD *)(v20 + 40) + 40LL);
      v27 = *(_DWORD *)(v26 + 24);
      if ( v27 == 11 || v27 == 35 )
      {
        v28 = *(_QWORD *)(v26 + 96);
        v29 = v18;
        v43 = *(_DWORD *)(v28 + 32);
        if ( v43 > 0x40 )
        {
          v41 = v20;
          v39 = sub_C444A0(v28 + 24);
          v29 = v18;
          v20 = v41;
          if ( v43 - v39 > 0x40 )
            goto LABEL_19;
          v30 = **(_QWORD **)(v28 + 24);
        }
        else
        {
          v30 = *(_QWORD *)(v28 + 24);
        }
        if ( v29 == v30 )
        {
          v44 = v20;
          v10 = sub_33DD210(*a1, v50, v48, (__int64)&v55, 0);
          if ( (_BYTE)v10 )
          {
            *(_QWORD *)a4 = v50;
            *(_DWORD *)(a4 + 8) = v48;
            v31 = *(_QWORD *)(v44 + 40);
            *(_QWORD *)a5 = *(_QWORD *)v31;
            *(_DWORD *)(a5 + 8) = *(_DWORD *)(v31 + 8);
            goto LABEL_21;
          }
        }
      }
    }
LABEL_19:
    if ( *(_DWORD *)(v19 + 24) == 190 )
    {
      v32 = *(_QWORD *)(*(_QWORD *)(v19 + 40) + 40LL);
      v33 = *(_DWORD *)(v32 + 24);
      if ( v33 == 35 || v33 == 11 )
      {
        v34 = *(_QWORD *)(v32 + 96);
        v35 = v18;
        v36 = *(_DWORD *)(v34 + 32);
        if ( v36 <= 0x40 )
        {
          v37 = *(_QWORD *)(v34 + 24);
LABEL_39:
          if ( v35 == v37 )
          {
            v10 = sub_33DD210(*a1, v47, v46, (__int64)&v55, 0);
            if ( (_BYTE)v10 )
            {
              *(_QWORD *)a4 = v47;
              *(_DWORD *)(a4 + 8) = v46;
              v38 = *(_QWORD *)(v19 + 40);
              *(_QWORD *)a5 = *(_QWORD *)v38;
              *(_DWORD *)(a5 + 8) = *(_DWORD *)(v38 + 8);
              goto LABEL_21;
            }
          }
          goto LABEL_20;
        }
        v49 = v35;
        v52 = v34;
        v40 = sub_C444A0(v34 + 24);
        v35 = v49;
        if ( v36 - v40 <= 0x40 )
        {
          v37 = **(_QWORD **)(v52 + 24);
          goto LABEL_39;
        }
      }
    }
LABEL_20:
    v10 = 0;
LABEL_21:
    if ( (unsigned int)v56 > 0x40 && v55 )
    {
      v51 = v10;
      j_j___libc_free_0_0(v55);
      return v51;
    }
    return v10;
  }
  if ( (unsigned __int16)(v8 - 17) <= 0xD3u )
  {
    LOWORD(v8) = word_4456580[v8 - 1];
    v13 = 0;
    goto LABEL_9;
  }
  LOWORD(v55) = v8;
  v56 = v9;
LABEL_4:
  if ( (_WORD)v8 == 1 || (unsigned __int16)(v8 - 504) <= 7u )
    BUG();
  v10 = 0;
  v11 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v8 - 16];
  if ( *(_DWORD *)(a2 + 24) == 187 )
    goto LABEL_11;
  return v10;
}
