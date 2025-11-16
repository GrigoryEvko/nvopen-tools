// Function: sub_358FB10
// Address: 0x358fb10
//
__int64 __fastcall sub_358FB10(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // r13
  __int64 v5; // rbx
  char v6; // r15
  unsigned int v7; // esi
  __int64 v8; // rcx
  __int64 v9; // r9
  int v10; // r11d
  _QWORD *v11; // rdx
  __int64 v12; // r8
  _QWORD *v13; // rax
  __int64 v14; // rdi
  __int64 *v15; // rax
  _QWORD *v16; // rax
  __int64 v18; // rdx
  _QWORD *v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  void (__fastcall ***v22)(unsigned __int64 *, _QWORD, __int64); // rsi
  __int64 v23; // r12
  __int64 v24; // r14
  __int64 *v25; // rax
  int v26; // eax
  int v27; // edi
  int v28; // eax
  unsigned int v29; // eax
  __int64 v30; // rsi
  int v31; // r11d
  _QWORD *v32; // r10
  int v33; // eax
  int v34; // eax
  unsigned int v35; // r15d
  int v36; // r10d
  __int64 v37; // rsi
  __int64 v38; // [rsp+8h] [rbp-88h]
  __int64 v39; // [rsp+8h] [rbp-88h]
  __int64 v40; // [rsp+10h] [rbp-80h]
  unsigned __int8 v43; // [rsp+2Fh] [rbp-61h]
  __int64 v44; // [rsp+38h] [rbp-58h] BYREF
  _QWORD v45[2]; // [rsp+40h] [rbp-50h] BYREF
  char v46; // [rsp+50h] [rbp-40h]

  v3 = a2 + 40;
  v5 = a2[41];
  v43 = *(_DWORD *)(a3 + 16) != 0;
  if ( (__int64 *)v5 != a2 + 40 )
  {
    v6 = 0;
    v40 = a1 + 40;
    while ( 1 )
    {
      while ( 1 )
      {
        sub_3586670((__int64)v45, (void (__fastcall ***)(unsigned __int64 *, _QWORD, __int64))a1, v5);
        if ( (v46 & 1) == 0 )
          break;
        v5 = *(_QWORD *)(v5 + 8);
        if ( v3 == (__int64 *)v5 )
          goto LABEL_15;
      }
      v7 = *(_DWORD *)(a1 + 64);
      v8 = v45[0];
      if ( !v7 )
        break;
      v9 = *(_QWORD *)(a1 + 48);
      v10 = 1;
      v11 = 0;
      v12 = (v7 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v13 = (_QWORD *)(v9 + 16 * v12);
      v14 = *v13;
      if ( *v13 != v5 )
      {
        while ( v14 != -4096 )
        {
          if ( v14 == -8192 && !v11 )
            v11 = v13;
          v12 = (v7 - 1) & (v10 + (_DWORD)v12);
          v13 = (_QWORD *)(v9 + 16LL * (unsigned int)v12);
          v14 = *v13;
          if ( *v13 == v5 )
            goto LABEL_7;
          ++v10;
        }
        if ( !v11 )
          v11 = v13;
        v26 = *(_DWORD *)(a1 + 56);
        ++*(_QWORD *)(a1 + 40);
        v27 = v26 + 1;
        if ( 4 * (v26 + 1) < 3 * v7 )
        {
          v12 = v7 >> 3;
          if ( v7 - *(_DWORD *)(a1 + 60) - v27 <= (unsigned int)v12 )
          {
            v39 = v8;
            sub_2E3E470(v40, v7);
            v33 = *(_DWORD *)(a1 + 64);
            if ( !v33 )
            {
LABEL_63:
              ++*(_DWORD *)(a1 + 56);
              BUG();
            }
            v34 = v33 - 1;
            v12 = *(_QWORD *)(a1 + 48);
            v9 = 0;
            v35 = v34 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
            v36 = 1;
            v27 = *(_DWORD *)(a1 + 56) + 1;
            v8 = v39;
            v11 = (_QWORD *)(v12 + 16LL * v35);
            v37 = *v11;
            if ( *v11 != v5 )
            {
              while ( v37 != -4096 )
              {
                if ( v37 == -8192 && !v9 )
                  v9 = (__int64)v11;
                v35 = v34 & (v36 + v35);
                v11 = (_QWORD *)(v12 + 16LL * v35);
                v37 = *v11;
                if ( *v11 == v5 )
                  goto LABEL_36;
                ++v36;
              }
              if ( v9 )
                v11 = (_QWORD *)v9;
            }
          }
          goto LABEL_36;
        }
LABEL_40:
        v38 = v8;
        sub_2E3E470(v40, 2 * v7);
        v28 = *(_DWORD *)(a1 + 64);
        if ( !v28 )
          goto LABEL_63;
        v9 = (unsigned int)(v28 - 1);
        v12 = *(_QWORD *)(a1 + 48);
        v29 = v9 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
        v27 = *(_DWORD *)(a1 + 56) + 1;
        v8 = v38;
        v11 = (_QWORD *)(v12 + 16LL * v29);
        v30 = *v11;
        if ( *v11 != v5 )
        {
          v31 = 1;
          v32 = 0;
          while ( v30 != -4096 )
          {
            if ( v30 == -8192 && !v32 )
              v32 = v11;
            v29 = v9 & (v31 + v29);
            v11 = (_QWORD *)(v12 + 16LL * v29);
            v30 = *v11;
            if ( *v11 == v5 )
              goto LABEL_36;
            ++v31;
          }
          if ( v32 )
            v11 = v32;
        }
LABEL_36:
        *(_DWORD *)(a1 + 56) = v27;
        if ( *v11 != -4096 )
          --*(_DWORD *)(a1 + 60);
        *v11 = v5;
        v15 = v11 + 1;
        v11[1] = 0;
        goto LABEL_8;
      }
LABEL_7:
      v15 = v13 + 1;
LABEL_8:
      *v15 = v8;
      if ( !*(_BYTE *)(a1 + 132) )
        goto LABEL_19;
      v16 = *(_QWORD **)(a1 + 112);
      v8 = *(unsigned int *)(a1 + 124);
      v11 = &v16[v8];
      if ( v16 != v11 )
      {
        while ( *v16 != v5 )
        {
          if ( v11 == ++v16 )
            goto LABEL_18;
        }
        v6 = 1;
        goto LABEL_14;
      }
LABEL_18:
      if ( (unsigned int)v8 < *(_DWORD *)(a1 + 120) )
      {
        v6 = 1;
        *(_DWORD *)(a1 + 124) = v8 + 1;
        *v11 = v5;
        ++*(_QWORD *)(a1 + 104);
      }
      else
      {
LABEL_19:
        v6 = 1;
        sub_C8CC70(a1 + 104, v5, (__int64)v11, v8, v12, v9);
      }
LABEL_14:
      v5 = *(_QWORD *)(v5 + 8);
      if ( v3 == (__int64 *)v5 )
      {
LABEL_15:
        v43 |= v6;
        goto LABEL_16;
      }
    }
    ++*(_QWORD *)(a1 + 40);
    goto LABEL_40;
  }
LABEL_16:
  if ( v43 )
  {
    sub_B2F4C0(*a2, *(_QWORD *)(*(_QWORD *)(a1 + 1200) + 64LL) + 1LL, 0, (_BYTE *)a3);
    if ( !LOBYTE(qword_500BA28[8]) )
    {
      nullsub_1894();
      sub_3588E20(a1, (__int64)a2);
    }
    sub_358ABB0(a1, (__int64)a2);
    sub_358F5F0(a1, (__int64)a2, v18, v19, v20, v21);
    if ( LOBYTE(qword_500BA28[8]) )
    {
      v22 = (void (__fastcall ***)(unsigned __int64 *, _QWORD, __int64))a1;
      v23 = a1 + 40;
      v44 = a2[41];
      sub_3586670((__int64)v45, v22, v44);
      if ( *sub_3588500(v23, &v44) )
      {
        v24 = *a2;
        v25 = sub_3588500(v23, &v44);
        sub_B2F4C0(v24, *v25, 0, (_BYTE *)a3);
      }
    }
  }
  return v43;
}
