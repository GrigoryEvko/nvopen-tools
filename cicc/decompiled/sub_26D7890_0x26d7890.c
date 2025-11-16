// Function: sub_26D7890
// Address: 0x26d7890
//
void __fastcall sub_26D7890(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  unsigned __int64 v4; // r9
  int v5; // r11d
  __int64 *v6; // rdi
  __int64 v7; // rcx
  __int64 *v8; // rdx
  __int64 v9; // r8
  __int64 v10; // rax
  unsigned int v11; // esi
  __int64 v12; // r8
  int v13; // ecx
  __int64 v14; // r13
  __int64 v15; // rdi
  int v16; // r15d
  __int64 *v17; // r9
  unsigned int v18; // ecx
  __int64 *v19; // rdx
  __int64 v20; // r11
  __int64 v21; // rdx
  __int64 v22; // r14
  __int64 v23; // rax
  unsigned int v24; // esi
  int v25; // ecx
  int v26; // edi
  int v27; // ecx
  __int64 v28; // rsi
  __int64 v29; // rdx
  __int64 v30; // rcx
  unsigned int v31; // eax
  __int64 v32; // rax
  unsigned int v33; // edx
  __int64 *v34; // rdi
  unsigned int v35; // eax
  __int64 v36; // r13
  __int64 v37; // rax
  unsigned __int64 v38; // rdx
  __int64 v39; // r15
  __int64 v40; // rbx
  __int64 v41; // r15
  __int64 v42; // r13
  __int64 v43; // rdx
  __int64 *v44; // rcx
  unsigned __int64 i; // rax
  __int64 v46; // rcx
  __int64 v48; // [rsp+8h] [rbp-108h]
  __int64 v49; // [rsp+20h] [rbp-F0h]
  __int64 v50; // [rsp+28h] [rbp-E8h]
  __int64 v51; // [rsp+38h] [rbp-D8h] BYREF
  __int64 *v52; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v53; // [rsp+48h] [rbp-C8h]
  _BYTE v54[64]; // [rsp+50h] [rbp-C0h] BYREF
  __int64 *v55; // [rsp+90h] [rbp-80h] BYREF
  __int64 v56; // [rsp+98h] [rbp-78h]
  _QWORD v57[14]; // [rsp+A0h] [rbp-70h] BYREF

  v2 = *(_QWORD *)(a2 + 80);
  v52 = (__int64 *)v54;
  v53 = 0x800000000LL;
  v50 = a2 + 72;
  if ( v2 != a2 + 72 )
  {
    v3 = a1;
    v48 = a1 + 968;
    while ( 1 )
    {
      v10 = v2 - 24;
      if ( !v2 )
        v10 = 0;
      v11 = *(_DWORD *)(v3 + 992);
      v51 = v10;
      if ( !v11 )
        break;
      v4 = *(_QWORD *)(v3 + 976);
      v5 = 1;
      v6 = 0;
      LODWORD(v7) = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v8 = (__int64 *)(v4 + 16LL * (unsigned int)v7);
      v9 = *v8;
      if ( v10 != *v8 )
      {
        while ( v9 != -4096 )
        {
          if ( v6 || v9 != -8192 )
            v8 = v6;
          v7 = (v11 - 1) & ((_DWORD)v7 + v5);
          v9 = *(_QWORD *)(v4 + 16 * v7);
          if ( v10 == v9 )
            goto LABEL_4;
          ++v5;
          v6 = v8;
          v8 = (__int64 *)(v4 + 16 * v7);
        }
        v27 = *(_DWORD *)(v3 + 984);
        if ( !v6 )
          v6 = v8;
        ++*(_QWORD *)(v3 + 968);
        v13 = v27 + 1;
        v55 = v6;
        if ( 4 * v13 < 3 * v11 )
        {
          v12 = v11 >> 3;
          if ( v11 - *(_DWORD *)(v3 + 988) - v13 > (unsigned int)v12 )
            goto LABEL_49;
          goto LABEL_10;
        }
LABEL_9:
        v11 *= 2;
LABEL_10:
        sub_1059000(v48, v11);
        sub_26CE030(v48, &v51, &v55);
        v10 = v51;
        v13 = *(_DWORD *)(v3 + 984) + 1;
        v6 = v55;
LABEL_49:
        *(_DWORD *)(v3 + 984) = v13;
        if ( *v6 != -4096 )
          --*(_DWORD *)(v3 + 988);
        *v6 = v10;
        v28 = v51;
        v6[1] = v51;
        v29 = *(_QWORD *)(v3 + 1000);
        LODWORD(v53) = 0;
        if ( v28 )
        {
          v30 = (unsigned int)(*(_DWORD *)(v28 + 44) + 1);
          v31 = *(_DWORD *)(v28 + 44) + 1;
        }
        else
        {
          v30 = 0;
          v31 = 0;
        }
        if ( v31 < *(_DWORD *)(v29 + 32) && (v32 = *(_QWORD *)(*(_QWORD *)(v29 + 24) + 8 * v30)) != 0 )
        {
          v49 = v3;
          v33 = 0;
          v55 = v57;
          v34 = v57;
          v57[0] = v32;
          v56 = 0x800000001LL;
          v35 = 1;
          while ( 1 )
          {
            v36 = v34[v35 - 1];
            LODWORD(v56) = v35 - 1;
            v37 = v33;
            v38 = v33 + 1LL;
            v39 = *(_QWORD *)v36;
            if ( v38 > HIDWORD(v53) )
            {
              sub_C8D5F0((__int64)&v52, v54, v38, 8u, v12, v4);
              v37 = (unsigned int)v53;
            }
            v52[v37] = v39;
            v40 = *(_QWORD *)(v36 + 24);
            v41 = *(unsigned int *)(v36 + 32);
            v42 = 8 * v41;
            v43 = (unsigned int)v56;
            LODWORD(v53) = v53 + 1;
            v4 = v41 + (unsigned int)v56;
            if ( v4 > HIDWORD(v56) )
            {
              sub_C8D5F0((__int64)&v55, v57, v41 + (unsigned int)v56, 8u, v12, v4);
              v43 = (unsigned int)v56;
            }
            v34 = v55;
            v44 = &v55[v43];
            if ( v42 )
            {
              for ( i = 0; i != v42; i += 8LL )
                v44[i / 8] = *(_QWORD *)(v40 + i);
              v34 = v55;
              LODWORD(v43) = v56;
            }
            LODWORD(v56) = v41 + v43;
            v35 = v41 + v43;
            if ( !((_DWORD)v41 + (_DWORD)v43) )
              break;
            v33 = v53;
          }
          v3 = v49;
          if ( v34 != v57 )
            _libc_free((unsigned __int64)v34);
          v28 = v51;
          v46 = (unsigned int)v53;
        }
        else
        {
          v46 = 0;
        }
        sub_26D7400(v3, v28, v52, v46, *(_QWORD *)(v3 + 1008));
      }
LABEL_4:
      v2 = *(_QWORD *)(v2 + 8);
      if ( v50 == v2 )
      {
        v14 = *(_QWORD *)(a2 + 80);
        if ( v50 == v14 )
        {
LABEL_11:
          if ( v52 != (__int64 *)v54 )
            _libc_free((unsigned __int64)v52);
          return;
        }
        while ( 2 )
        {
          v23 = v14 - 24;
          if ( !v14 )
            v23 = 0;
          v24 = *(_DWORD *)(v3 + 992);
          v51 = v23;
          if ( !v24 )
          {
            ++*(_QWORD *)(v3 + 968);
            v55 = 0;
            goto LABEL_25;
          }
          v15 = *(_QWORD *)(v3 + 976);
          v16 = 1;
          v17 = 0;
          v18 = (v24 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v19 = (__int64 *)(v15 + 16LL * v18);
          v20 = *v19;
          if ( v23 == *v19 )
          {
LABEL_17:
            v21 = v19[1];
            goto LABEL_18;
          }
          while ( v20 != -4096 )
          {
            if ( !v17 && v20 == -8192 )
              v17 = v19;
            v18 = (v24 - 1) & (v16 + v18);
            v19 = (__int64 *)(v15 + 16LL * v18);
            v20 = *v19;
            if ( v23 == *v19 )
              goto LABEL_17;
            ++v16;
          }
          v26 = *(_DWORD *)(v3 + 984);
          if ( !v17 )
            v17 = v19;
          ++*(_QWORD *)(v3 + 968);
          v25 = v26 + 1;
          v55 = v17;
          if ( 4 * (v26 + 1) >= 3 * v24 )
          {
LABEL_25:
            v24 *= 2;
          }
          else if ( v24 - *(_DWORD *)(v3 + 988) - v25 > v24 >> 3 )
          {
            goto LABEL_37;
          }
          sub_1059000(v48, v24);
          sub_26CE030(v48, &v51, &v55);
          v23 = v51;
          v17 = v55;
          v25 = *(_DWORD *)(v3 + 984) + 1;
LABEL_37:
          *(_DWORD *)(v3 + 984) = v25;
          if ( *v17 != -4096 )
            --*(_DWORD *)(v3 + 988);
          *v17 = v23;
          v21 = 0;
          v17[1] = 0;
          v23 = v51;
LABEL_18:
          v55 = (__int64 *)v21;
          if ( v23 != v21 )
          {
            v22 = *sub_26CC460(v3 + 40, (__int64 *)&v55);
            *sub_26CC460(v3 + 40, &v51) = v22;
          }
          v14 = *(_QWORD *)(v14 + 8);
          if ( v2 == v14 )
            goto LABEL_11;
          continue;
        }
      }
    }
    ++*(_QWORD *)(v3 + 968);
    v55 = 0;
    goto LABEL_9;
  }
}
