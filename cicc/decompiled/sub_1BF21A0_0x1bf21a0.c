// Function: sub_1BF21A0
// Address: 0x1bf21a0
//
__int64 __fastcall sub_1BF21A0(__int64 *a1)
{
  __int64 *v2; // rbx
  __int64 v3; // rax
  __int64 v4; // r14
  unsigned int v5; // r15d
  __int64 *v6; // rbx
  unsigned __int64 v7; // rax
  _QWORD *v8; // r15
  char *v9; // rax
  _QWORD *v10; // r14
  _QWORD *v11; // r15
  _QWORD *v12; // rdi
  __int64 v14; // rax
  int v15; // ecx
  __int64 v16; // rdi
  __int64 v17; // r8
  int v18; // ecx
  unsigned int v19; // esi
  __int64 *v20; // rax
  __int64 v21; // r10
  __int64 v22; // rax
  __int64 v23; // rsi
  unsigned int v24; // edx
  __int64 *v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rax
  _QWORD *v28; // r14
  char *v29; // rax
  _QWORD *v30; // r15
  _QWORD *v31; // r14
  _QWORD *v32; // rdi
  __int64 v33; // rbx
  _QWORD *v34; // r13
  char *v35; // rax
  _QWORD *v36; // rbx
  _QWORD *v37; // r12
  _QWORD *v38; // rdi
  __int64 v39; // rax
  __int64 v40; // r13
  int v41; // eax
  int v42; // r9d
  int v43; // eax
  int v44; // r9d
  unsigned __int64 v45; // [rsp+8h] [rbp-228h]
  __int64 v46; // [rsp+8h] [rbp-228h]
  char v47; // [rsp+17h] [rbp-219h]
  __int64 *v48; // [rsp+18h] [rbp-218h]
  _QWORD v49[11]; // [rsp+20h] [rbp-210h] BYREF
  _QWORD *v50; // [rsp+78h] [rbp-1B8h]
  unsigned int v51; // [rsp+80h] [rbp-1B0h]
  _BYTE v52[424]; // [rsp+88h] [rbp-1A8h] BYREF

  v2 = (__int64 *)a1[7];
  v3 = sub_15E0530(*v2);
  v47 = 1;
  if ( !sub_1602790(v3) )
  {
    v39 = sub_15E0530(*v2);
    v40 = sub_16033E0(v39);
    if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v40 + 32LL))(
           v40,
           "loop-vectorize",
           14)
      || (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v40 + 40LL))(
           v40,
           "loop-vectorize",
           14)
      || (v47 = (*(__int64 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v40 + 24LL))(
                  v40,
                  "loop-vectorize",
                  14)) != 0 )
    {
      v47 = 1;
    }
  }
  v4 = *a1;
  v5 = 1;
  v6 = *(__int64 **)(*a1 + 32);
  v48 = *(__int64 **)(*a1 + 40);
  if ( v6 != v48 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v7 = sub_157EBA0(*v6);
        if ( *(_BYTE *)(v7 + 16) != 26 )
        {
          v8 = (_QWORD *)a1[7];
          v9 = sub_1BF18B0(a1[58]);
          sub_1BF1750((__int64)v49, (__int64)v9, (__int64)"CFGNotUnderstood", 16, v4, 0);
          sub_15CAB20((__int64)v49, "loop control flow is not understood by vectorizer", 0x31u);
          sub_143AA50(v8, (__int64)v49);
          v10 = v50;
          v49[0] = &unk_49ECF68;
          v11 = &v50[11 * v51];
          if ( v50 != v11 )
          {
            do
            {
              v11 -= 11;
              v12 = (_QWORD *)v11[4];
              if ( v12 != v11 + 6 )
                j_j___libc_free_0(v12, v11[6] + 1LL);
              if ( (_QWORD *)*v11 != v11 + 2 )
                j_j___libc_free_0(*v11, v11[2] + 1LL);
            }
            while ( v10 != v11 );
            v11 = v50;
          }
          if ( v11 != (_QWORD *)v52 )
            _libc_free((unsigned __int64)v11);
LABEL_16:
          if ( !v47 )
            return 0;
          goto LABEL_17;
        }
        if ( (*(_DWORD *)(v7 + 20) & 0xFFFFFFF) == 3 )
          break;
LABEL_5:
        if ( v48 == ++v6 )
          goto LABEL_18;
      }
      v45 = v7;
      if ( sub_13FC1A0(v4, *(_QWORD *)(v7 - 72)) )
        break;
      v14 = a1[1];
      v15 = *(_DWORD *)(v14 + 24);
      if ( v15 )
      {
        v16 = *(_QWORD *)(v45 - 24);
        v17 = *(_QWORD *)(v14 + 8);
        v18 = v15 - 1;
        v19 = v18 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v20 = (__int64 *)(v17 + 16LL * v19);
        v21 = *v20;
        if ( v16 == *v20 )
        {
LABEL_23:
          v22 = v20[1];
          if ( v22 && v16 == **(_QWORD **)(v22 + 32) )
            break;
        }
        else
        {
          v41 = 1;
          while ( v21 != -8 )
          {
            v42 = v41 + 1;
            v19 = v18 & (v41 + v19);
            v20 = (__int64 *)(v17 + 16LL * v19);
            v21 = *v20;
            if ( v16 == *v20 )
              goto LABEL_23;
            v41 = v42;
          }
        }
        v23 = *(_QWORD *)(v45 - 48);
        v24 = v18 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v25 = (__int64 *)(v17 + 16LL * v24);
        v26 = *v25;
        if ( *v25 == v23 )
        {
LABEL_26:
          v27 = v25[1];
          if ( v27 && **(_QWORD **)(v27 + 32) == v23 )
            break;
        }
        else
        {
          v43 = 1;
          while ( v26 != -8 )
          {
            v44 = v43 + 1;
            v24 = v18 & (v43 + v24);
            v25 = (__int64 *)(v17 + 16LL * v24);
            v26 = *v25;
            if ( *v25 == v23 )
              goto LABEL_26;
            v43 = v44;
          }
        }
      }
      v28 = (_QWORD *)a1[7];
      v46 = *a1;
      v29 = sub_1BF18B0(a1[58]);
      sub_1BF1750((__int64)v49, (__int64)v29, (__int64)"CFGNotUnderstood", 16, v46, 0);
      sub_15CAB20((__int64)v49, "loop control flow is not understood by vectorizer", 0x31u);
      sub_143AA50(v28, (__int64)v49);
      v30 = v50;
      v49[0] = &unk_49ECF68;
      v31 = &v50[11 * v51];
      if ( v50 != v31 )
      {
        do
        {
          v31 -= 11;
          v32 = (_QWORD *)v31[4];
          if ( v32 != v31 + 6 )
            j_j___libc_free_0(v32, v31[6] + 1LL);
          if ( (_QWORD *)*v31 != v31 + 2 )
            j_j___libc_free_0(*v31, v31[2] + 1LL);
        }
        while ( v30 != v31 );
        v31 = v50;
      }
      if ( v31 == (_QWORD *)v52 )
        goto LABEL_16;
      _libc_free((unsigned __int64)v31);
      if ( !v47 )
        return 0;
LABEL_17:
      v4 = *a1;
      v5 = 0;
      if ( v48 == ++v6 )
        goto LABEL_18;
    }
    v4 = *a1;
    goto LABEL_5;
  }
LABEL_18:
  if ( !sub_1BF0FA0(v4, v4) )
  {
    v33 = *a1;
    v34 = (_QWORD *)a1[7];
    v35 = sub_1BF18B0(a1[58]);
    sub_1BF1750((__int64)v49, (__int64)v35, (__int64)"CFGNotUnderstood", 16, v33, 0);
    sub_15CAB20((__int64)v49, "loop control flow is not understood by vectorizer", 0x31u);
    sub_143AA50(v34, (__int64)v49);
    v36 = v50;
    v49[0] = &unk_49ECF68;
    v37 = &v50[11 * v51];
    if ( v50 != v37 )
    {
      do
      {
        v37 -= 11;
        v38 = (_QWORD *)v37[4];
        if ( v38 != v37 + 6 )
          j_j___libc_free_0(v38, v37[6] + 1LL);
        if ( (_QWORD *)*v37 != v37 + 2 )
          j_j___libc_free_0(*v37, v37[2] + 1LL);
      }
      while ( v36 != v37 );
      v37 = v50;
    }
    if ( v37 == (_QWORD *)v52 )
    {
      return 0;
    }
    else
    {
      v5 = 0;
      _libc_free((unsigned __int64)v37);
    }
  }
  return v5;
}
