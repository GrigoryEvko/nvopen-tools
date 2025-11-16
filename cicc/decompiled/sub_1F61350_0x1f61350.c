// Function: sub_1F61350
// Address: 0x1f61350
//
char __fastcall sub_1F61350(__int64 a1, __int64 a2, char a3)
{
  _QWORD **v3; // rax
  unsigned int v6; // r13d
  int v7; // r14d
  __int64 v8; // rcx
  __int64 v9; // r12
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned int v13; // esi
  __int64 v14; // r9
  __int64 v15; // r8
  __int64 *v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // rdx
  unsigned int v19; // ecx
  unsigned int v20; // esi
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 *v23; // rcx
  int v24; // edi
  int v25; // edx
  int v26; // edi
  int v27; // edi
  __int64 v28; // r9
  unsigned int v29; // esi
  __int64 v30; // r8
  int v31; // r10d
  __int64 *v32; // r11
  int v33; // edi
  int v34; // edi
  __int64 v35; // r9
  int v36; // r10d
  unsigned int v37; // esi
  __int64 v38; // r8
  int v40; // [rsp+8h] [rbp-48h]
  __int64 v41; // [rsp+8h] [rbp-48h]
  unsigned int v42; // [rsp+10h] [rbp-40h]
  __int64 v43; // [rsp+10h] [rbp-40h]

  LODWORD(v3) = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( (_DWORD)v3 )
  {
    v6 = 0;
    v7 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    while ( 1 )
    {
      while ( 1 )
      {
        v8 = (*(_BYTE *)(a2 + 23) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        v9 = *(_QWORD *)(v8 + 8LL * v6 + 24LL * *(unsigned int *)(a2 + 56) + 8);
        v10 = sub_157EBA0(v9);
        if ( *(_BYTE *)(v10 + 16) != 33 )
          break;
        v11 = *(_QWORD *)(*(_QWORD *)(v10 - 48) - 24LL);
        if ( (*(_BYTE *)(v11 + 23) & 0x40) != 0 )
          v3 = *(_QWORD ***)(v11 - 8);
        else
          v3 = (_QWORD **)(v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF));
        LOBYTE(v3) = **(_QWORD **)a1 == (_QWORD)*v3;
        if ( (_BYTE)v3 != a3 )
          goto LABEL_8;
LABEL_16:
        v19 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
        if ( v19 )
        {
          v20 = 0;
          v21 = 24LL * *(unsigned int *)(a2 + 56) + 8;
          while ( 1 )
          {
            v22 = a2 - 24LL * v19;
            if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
              v22 = *(_QWORD *)(a2 - 8);
            if ( v9 == *(_QWORD *)(v22 + v21) )
              break;
            ++v20;
            v21 += 8;
            if ( v19 == v20 )
              goto LABEL_26;
          }
          --v7;
          LOBYTE(v3) = sub_15F5350(a2, v20, 0);
        }
        else
        {
LABEL_26:
          --v7;
          LOBYTE(v3) = sub_15F5350(a2, 0xFFFFFFFF, 0);
        }
        if ( v7 == v6 )
          return (char)v3;
      }
      v12 = *(_QWORD *)(a1 + 8);
      v13 = *(_DWORD *)(v12 + 192);
      if ( !v13 )
        break;
      v14 = *(_QWORD *)(v12 + 176);
      v42 = ((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4);
      v15 = (v13 - 1) & v42;
      v16 = (__int64 *)(v14 + 16 * v15);
      v17 = *v16;
      if ( v9 != *v16 )
      {
        v40 = 1;
        v23 = 0;
        while ( v17 != -8 )
        {
          if ( v17 == -16 && !v23 )
            v23 = v16;
          LODWORD(v15) = (v13 - 1) & (v40 + v15);
          v16 = (__int64 *)(v14 + 16LL * (unsigned int)v15);
          v17 = *v16;
          if ( v9 == *v16 )
            goto LABEL_13;
          ++v40;
        }
        v24 = *(_DWORD *)(v12 + 184);
        if ( !v23 )
          v23 = v16;
        ++*(_QWORD *)(v12 + 168);
        v25 = v24 + 1;
        if ( 4 * (v24 + 1) < 3 * v13 )
        {
          if ( v13 - *(_DWORD *)(v12 + 188) - v25 <= v13 >> 3 )
          {
            v41 = v12;
            sub_14DDDA0(v12 + 168, v13);
            v12 = v41;
            v33 = *(_DWORD *)(v41 + 192);
            if ( !v33 )
            {
LABEL_64:
              ++*(_DWORD *)(v12 + 184);
              BUG();
            }
            v34 = v33 - 1;
            v35 = *(_QWORD *)(v41 + 176);
            v32 = 0;
            v36 = 1;
            v37 = v34 & v42;
            v25 = *(_DWORD *)(v41 + 184) + 1;
            v23 = (__int64 *)(v35 + 16LL * (v34 & v42));
            v38 = *v23;
            if ( v9 != *v23 )
            {
              while ( v38 != -8 )
              {
                if ( v38 == -16 && !v32 )
                  v32 = v23;
                v37 = v34 & (v36 + v37);
                v23 = (__int64 *)(v35 + 16LL * v37);
                v38 = *v23;
                if ( v9 == *v23 )
                  goto LABEL_35;
                ++v36;
              }
              goto LABEL_51;
            }
          }
          goto LABEL_35;
        }
LABEL_39:
        v43 = v12;
        sub_14DDDA0(v12 + 168, 2 * v13);
        v12 = v43;
        v26 = *(_DWORD *)(v43 + 192);
        if ( !v26 )
          goto LABEL_64;
        v27 = v26 - 1;
        v28 = *(_QWORD *)(v43 + 176);
        v29 = v27 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v25 = *(_DWORD *)(v43 + 184) + 1;
        v23 = (__int64 *)(v28 + 16LL * v29);
        v30 = *v23;
        if ( v9 != *v23 )
        {
          v31 = 1;
          v32 = 0;
          while ( v30 != -8 )
          {
            if ( !v32 && v30 == -16 )
              v32 = v23;
            v29 = v27 & (v31 + v29);
            v23 = (__int64 *)(v28 + 16LL * v29);
            v30 = *v23;
            if ( v9 == *v23 )
              goto LABEL_35;
            ++v31;
          }
LABEL_51:
          if ( v32 )
            v23 = v32;
        }
LABEL_35:
        *(_DWORD *)(v12 + 184) = v25;
        if ( *v23 != -8 )
          --*(_DWORD *)(v12 + 188);
        *v23 = v9;
        v3 = 0;
        v23[1] = 0;
LABEL_14:
        v3 = (_QWORD **)**v3;
        goto LABEL_15;
      }
LABEL_13:
      v18 = v16[1];
      v3 = (_QWORD **)(v18 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v18 & 4) != 0 || !v3 )
        goto LABEL_14;
LABEL_15:
      LOBYTE(v3) = **(_QWORD **)(a1 + 16) == (_QWORD)v3;
      if ( (_BYTE)v3 == a3 )
        goto LABEL_16;
LABEL_8:
      if ( v7 == ++v6 )
        return (char)v3;
    }
    ++*(_QWORD *)(v12 + 168);
    goto LABEL_39;
  }
  return (char)v3;
}
