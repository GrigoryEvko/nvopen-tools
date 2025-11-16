// Function: sub_123FA00
// Address: 0x123fa00
//
__int64 __fastcall sub_123FA00(__int64 a1, __int64 a2)
{
  unsigned int v4; // r12d
  __int64 v5; // r8
  __int64 v6; // r9
  unsigned int v8; // esi
  __int64 v9; // rcx
  unsigned int v10; // edx
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 **v13; // rax
  __int64 v14; // rax
  __int64 v15; // r12
  int v16; // eax
  unsigned __int64 v17; // rsi
  __int64 *v18; // rax
  int v19; // edi
  int v20; // edx
  __int64 *v21; // rdx
  int v22; // edi
  int v23; // edi
  unsigned int v24; // esi
  int v25; // r11d
  __int64 *v26; // r10
  int v27; // edi
  int v28; // edi
  int v29; // r11d
  unsigned int v30; // esi
  __int64 v31; // [rsp+0h] [rbp-80h]
  __int64 **v32; // [rsp+8h] [rbp-78h]
  int v33; // [rsp+8h] [rbp-78h]
  unsigned int v34; // [rsp+1Ch] [rbp-64h] BYREF
  __int64 v35[4]; // [rsp+20h] [rbp-60h] BYREF
  char v36; // [rsp+40h] [rbp-40h]
  char v37; // [rsp+41h] [rbp-3Fh]

  if ( *(_DWORD *)(a1 + 240) == 511 )
  {
    while ( 1 )
    {
      v4 = sub_122E830(a1, &v34, v35);
      if ( (_BYTE)v4 )
        return 1;
      if ( v34 != 38 )
      {
        sub_B99FD0(a2, v34, v35[0]);
        if ( v34 != 1 )
          goto LABEL_5;
        goto LABEL_12;
      }
      v8 = *(_DWORD *)(a1 + 920);
      if ( v8 )
      {
        v9 = v35[0];
        v6 = *(_QWORD *)(a1 + 904);
        v10 = (v8 - 1) & ((LODWORD(v35[0]) >> 9) ^ (LODWORD(v35[0]) >> 4));
        v5 = v6 + 40LL * v10;
        v11 = *(_QWORD *)v5;
        if ( v35[0] == *(_QWORD *)v5 )
        {
LABEL_9:
          v12 = *(unsigned int *)(v5 + 16);
          v13 = (__int64 **)(v5 + 8);
          v6 = v12 + 1;
          if ( *(unsigned int *)(v5 + 20) < (unsigned __int64)(v12 + 1) )
          {
            v31 = v5;
            v32 = (__int64 **)(v5 + 8);
            sub_C8D5F0(v5 + 8, (const void *)(v5 + 24), v12 + 1, 8u, v5, v6);
            v5 = v31;
            v13 = v32;
            v12 = *(unsigned int *)(v31 + 16);
          }
          goto LABEL_11;
        }
        v33 = 1;
        v18 = 0;
        while ( v11 != -4096 )
        {
          if ( v11 == -8192 && !v18 )
            v18 = (__int64 *)v5;
          v10 = (v8 - 1) & (v33 + v10);
          v6 = (unsigned int)(v33 + 1);
          v5 = *(_QWORD *)(a1 + 904) + 40LL * v10;
          v11 = *(_QWORD *)v5;
          if ( v35[0] == *(_QWORD *)v5 )
            goto LABEL_9;
          ++v33;
        }
        v19 = *(_DWORD *)(a1 + 912);
        if ( !v18 )
          v18 = (__int64 *)v5;
        ++*(_QWORD *)(a1 + 896);
        v20 = v19 + 1;
        if ( 4 * (v19 + 1) < 3 * v8 )
        {
          v5 = v8 >> 3;
          if ( v8 - *(_DWORD *)(a1 + 916) - v20 > (unsigned int)v5 )
            goto LABEL_24;
          sub_123F150(a1 + 896, v8);
          v27 = *(_DWORD *)(a1 + 920);
          if ( !v27 )
          {
LABEL_54:
            ++*(_DWORD *)(a1 + 912);
            BUG();
          }
          v9 = v35[0];
          v28 = v27 - 1;
          v26 = 0;
          v6 = *(_QWORD *)(a1 + 904);
          v29 = 1;
          v20 = *(_DWORD *)(a1 + 912) + 1;
          v30 = v28 & ((LODWORD(v35[0]) >> 9) ^ (LODWORD(v35[0]) >> 4));
          v18 = (__int64 *)(v6 + 40LL * v30);
          v5 = *v18;
          if ( *v18 == v35[0] )
            goto LABEL_24;
          while ( v5 != -4096 )
          {
            if ( v5 == -8192 && !v26 )
              v26 = v18;
            v30 = v28 & (v29 + v30);
            v18 = (__int64 *)(v6 + 40LL * v30);
            v5 = *v18;
            if ( v35[0] == *v18 )
              goto LABEL_24;
            ++v29;
          }
          goto LABEL_33;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 896);
      }
      sub_123F150(a1 + 896, 2 * v8);
      v22 = *(_DWORD *)(a1 + 920);
      if ( !v22 )
        goto LABEL_54;
      v9 = v35[0];
      v23 = v22 - 1;
      v6 = *(_QWORD *)(a1 + 904);
      v20 = *(_DWORD *)(a1 + 912) + 1;
      v24 = v23 & ((LODWORD(v35[0]) >> 9) ^ (LODWORD(v35[0]) >> 4));
      v18 = (__int64 *)(v6 + 40LL * v24);
      v5 = *v18;
      if ( *v18 == v35[0] )
        goto LABEL_24;
      v25 = 1;
      v26 = 0;
      while ( v5 != -4096 )
      {
        if ( !v26 && v5 == -8192 )
          v26 = v18;
        v24 = v23 & (v25 + v24);
        v18 = (__int64 *)(v6 + 40LL * v24);
        v5 = *v18;
        if ( v35[0] == *v18 )
          goto LABEL_24;
        ++v25;
      }
LABEL_33:
      if ( v26 )
        v18 = v26;
LABEL_24:
      *(_DWORD *)(a1 + 912) = v20;
      if ( *v18 != -4096 )
        --*(_DWORD *)(a1 + 916);
      v21 = v18 + 3;
      *v18 = v9;
      v13 = (__int64 **)(v18 + 1);
      *v13 = v21;
      v12 = 0;
      v13[1] = (__int64 *)0x200000000LL;
LABEL_11:
      (*v13)[v12] = a2;
      ++*((_DWORD *)v13 + 2);
      if ( v34 != 1 )
      {
LABEL_5:
        if ( *(_DWORD *)(a1 + 240) != 4 )
          return v4;
        goto LABEL_15;
      }
LABEL_12:
      v14 = *(unsigned int *)(a1 + 376);
      if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 380) )
      {
        sub_C8D5F0(a1 + 368, (const void *)(a1 + 384), v14 + 1, 8u, v5, v6);
        v14 = *(unsigned int *)(a1 + 376);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 368) + 8 * v14) = a2;
      ++*(_DWORD *)(a1 + 376);
      if ( *(_DWORD *)(a1 + 240) != 4 )
        return v4;
LABEL_15:
      v15 = a1 + 176;
      v16 = sub_1205200(a1 + 176);
      *(_DWORD *)(a1 + 240) = v16;
      if ( v16 != 511 )
        goto LABEL_16;
    }
  }
  v15 = a1 + 176;
LABEL_16:
  v37 = 1;
  v17 = *(_QWORD *)(a1 + 232);
  v36 = 3;
  v35[0] = (__int64)"expected metadata after comma";
  sub_11FD800(v15, v17, (__int64)v35, 1);
  return 1;
}
