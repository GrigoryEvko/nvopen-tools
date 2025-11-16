// Function: sub_1D06C30
// Address: 0x1d06c30
//
__int64 __fastcall sub_1D06C30(__int64 a1, __int64 a2)
{
  __int64 v4; // r15
  __int64 v5; // r14
  unsigned __int64 v6; // r13
  int v7; // eax
  unsigned __int64 v8; // r8
  unsigned int v9; // eax
  __int64 v10; // rdi
  char *(__fastcall *v11)(__int64, __int64); // rax
  int v12; // eax
  _BYTE *v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r13
  __int64 result; // rax
  unsigned __int64 v18; // rdx
  char v19; // al
  _BYTE *v20; // rsi
  __int64 v21; // rdi
  unsigned int *v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rax
  unsigned int v25; // esi
  __int64 v26; // r14
  __int64 v27; // rdi
  unsigned int v28; // ecx
  __int64 *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 *v32; // r10
  int v33; // r11d
  int v34; // ecx
  int v35; // edx
  int v36; // r8d
  int v37; // r8d
  __int64 v38; // r9
  unsigned int v39; // esi
  __int64 v40; // r11
  __int64 *v41; // rcx
  int v42; // edi
  int v43; // edi
  int v44; // edi
  __int64 v45; // r8
  __int64 v46; // r15
  __int64 v47; // r10
  int v48; // esi
  unsigned __int64 v49; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v50[14]; // [rsp+18h] [rbp-38h] BYREF

  v4 = *(_QWORD *)(a2 + 32);
  v5 = v4 + 16LL * *(unsigned int *)(a2 + 40);
  while ( v5 != v4 )
  {
    v6 = *(_QWORD *)v4 & 0xFFFFFFFFFFFFFFF8LL;
    v49 = v6;
    v7 = *(_DWORD *)(v6 + 212) - 1;
    *(_DWORD *)(v6 + 212) = v7;
    if ( *(_BYTE *)(a1 + 664) )
    {
      if ( (*(_BYTE *)(a2 + 236) & 2) == 0 )
        sub_1F01F70(a2);
      sub_1F020C0(v6, (unsigned int)(*(_DWORD *)(a2 + 244) + *(_DWORD *)(v4 + 12)));
      v6 = v49;
      v7 = *(_DWORD *)(v49 + 212);
    }
    if ( v7 || v6 == a1 + 72 )
      goto LABEL_21;
    *(_BYTE *)(v6 + 229) |= 2u;
    v8 = v6;
    if ( (*(_BYTE *)(v6 + 236) & 2) == 0 )
    {
      sub_1F01F70(v6);
      v8 = v49;
    }
    v9 = *(_DWORD *)(v6 + 244);
    if ( v9 < *(_DWORD *)(a1 + 716) )
      *(_DWORD *)(a1 + 716) = v9;
    v10 = *(_QWORD *)(a1 + 672);
    if ( !byte_4FC13A0 && *(_BYTE *)(v10 + 12) )
    {
      if ( !(*(unsigned __int8 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v10 + 80LL))(v10, v8) )
      {
        v18 = v49;
        v19 = *(_BYTE *)(v49 + 229);
        if ( (v19 & 1) == 0 )
        {
          *(_BYTE *)(v49 + 229) = v19 | 1;
          v20 = *(_BYTE **)(a1 + 688);
          if ( v20 == *(_BYTE **)(a1 + 696) )
          {
            sub_1CFD630(a1 + 680, v20, &v49);
          }
          else
          {
            if ( v20 )
            {
              *(_QWORD *)v20 = v18;
              v20 = *(_BYTE **)(a1 + 688);
            }
            *(_QWORD *)(a1 + 688) = v20 + 8;
          }
        }
        goto LABEL_21;
      }
      v10 = *(_QWORD *)(a1 + 672);
      v8 = v49;
    }
    v11 = *(char *(__fastcall **)(__int64, __int64))(*(_QWORD *)v10 + 88LL);
    if ( v11 == sub_1D047D0 )
    {
      *(_QWORD *)v50 = v8;
      v12 = *(_DWORD *)(v10 + 40) + 1;
      *(_DWORD *)(v10 + 40) = v12;
      *(_DWORD *)(v8 + 196) = v12;
      v13 = *(_BYTE **)(v10 + 24);
      if ( v13 == *(_BYTE **)(v10 + 32) )
      {
        sub_1CFD630(v10 + 16, v13, v50);
      }
      else
      {
        if ( v13 )
        {
          *(_QWORD *)v13 = v8;
          v13 = *(_BYTE **)(v10 + 24);
        }
        *(_QWORD *)(v10 + 24) = v13 + 8;
      }
    }
    else
    {
      v11(v10, v8);
    }
LABEL_21:
    if ( (*(_QWORD *)v4 & 6) == 0 )
    {
      v14 = *(unsigned int *)(v4 + 8);
      if ( (_DWORD)v14 )
      {
        *(_QWORD *)(*(_QWORD *)(a1 + 728) + 8 * v14) = *(_QWORD *)v4 & 0xFFFFFFFFFFFFFFF8LL;
        v15 = *(_QWORD *)(a1 + 736);
        if ( !*(_QWORD *)(v15 + 8LL * *(unsigned int *)(v4 + 8)) )
        {
          ++*(_DWORD *)(a1 + 724);
          *(_QWORD *)(v15 + 8LL * *(unsigned int *)(v4 + 8)) = a2;
        }
      }
    }
    v4 += 16;
  }
  v16 = *(unsigned int *)(*(_QWORD *)(a1 + 24) + 16LL);
  result = *(_QWORD *)(a1 + 728);
  if ( !*(_QWORD *)(result + 8 * v16) )
  {
    v21 = *(_QWORD *)a2;
    while ( 1 )
    {
      if ( !v21 )
        return result;
      if ( *(__int16 *)(v21 + 24) < 0 )
      {
        v23 = *(_QWORD *)(a1 + 16);
        if ( *(_DWORD *)(v23 + 40) == ~*(__int16 *)(v21 + 24) )
          break;
      }
      result = *(unsigned int *)(v21 + 56);
      if ( (_DWORD)result )
      {
        v22 = (unsigned int *)(*(_QWORD *)(v21 + 32) + 40LL * (unsigned int)(result - 1));
        v21 = *(_QWORD *)v22;
        result = *(_QWORD *)(*(_QWORD *)v22 + 40LL) + 16LL * v22[2];
        if ( *(_BYTE *)result == 111 )
          continue;
      }
      return result;
    }
    LODWORD(v49) = 0;
    v50[0] = 0;
    v24 = sub_1D00CF0(v21, (unsigned int *)&v49, v50, v23);
    v25 = *(_DWORD *)(a1 + 936);
    v26 = *(_QWORD *)(a1 + 48) + 272LL * *(int *)(v24 + 28);
    if ( v25 )
    {
      v27 = *(_QWORD *)(a1 + 920);
      v28 = (v25 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v29 = (__int64 *)(v27 + 16LL * v28);
      v30 = *v29;
      if ( v26 == *v29 )
      {
LABEL_42:
        v29[1] = a2;
        v31 = *(_QWORD *)(a1 + 728);
        ++*(_DWORD *)(a1 + 724);
        *(_QWORD *)(v31 + 8 * v16) = v26;
        result = *(_QWORD *)(a1 + 736);
        *(_QWORD *)(result + 8 * v16) = a2;
        return result;
      }
      v32 = 0;
      v33 = 1;
      while ( v30 != -8 )
      {
        if ( !v32 && v30 == -16 )
          v32 = v29;
        v28 = (v25 - 1) & (v33 + v28);
        v29 = (__int64 *)(v27 + 16LL * v28);
        v30 = *v29;
        if ( v26 == *v29 )
          goto LABEL_42;
        ++v33;
      }
      v34 = *(_DWORD *)(a1 + 928);
      if ( v32 )
        v29 = v32;
      ++*(_QWORD *)(a1 + 912);
      v35 = v34 + 1;
      if ( 4 * (v34 + 1) < 3 * v25 )
      {
        if ( v25 - *(_DWORD *)(a1 + 932) - v35 > v25 >> 3 )
        {
LABEL_51:
          *(_DWORD *)(a1 + 928) = v35;
          if ( *v29 != -8 )
            --*(_DWORD *)(a1 + 932);
          *v29 = v26;
          v29[1] = 0;
          goto LABEL_42;
        }
        sub_1D06A70(a1 + 912, v25);
        v43 = *(_DWORD *)(a1 + 936);
        if ( v43 )
        {
          v44 = v43 - 1;
          v45 = *(_QWORD *)(a1 + 920);
          LODWORD(v46) = v44 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
          v35 = *(_DWORD *)(a1 + 928) + 1;
          v29 = (__int64 *)(v45 + 16LL * (unsigned int)v46);
          v47 = *v29;
          if ( v26 == *v29 )
            goto LABEL_51;
          v41 = (__int64 *)(v45 + 16LL * (unsigned int)v46);
          v48 = 1;
          v29 = 0;
          while ( v47 != -8 )
          {
            if ( v47 == -16 && !v29 )
              v29 = v41;
            v46 = v44 & (unsigned int)(v46 + v48);
            v41 = (__int64 *)(v45 + 16 * v46);
            v47 = *v41;
            if ( v26 == *v41 )
              goto LABEL_70;
            ++v48;
          }
          goto LABEL_59;
        }
        goto LABEL_82;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 912);
    }
    sub_1D06A70(a1 + 912, 2 * v25);
    v36 = *(_DWORD *)(a1 + 936);
    if ( v36 )
    {
      v37 = v36 - 1;
      v38 = *(_QWORD *)(a1 + 920);
      v35 = *(_DWORD *)(a1 + 928) + 1;
      v39 = v37 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v29 = (__int64 *)(v38 + 16LL * v39);
      v40 = *v29;
      if ( v26 == *v29 )
        goto LABEL_51;
      v41 = (__int64 *)(v38 + 16LL * (v37 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4))));
      v42 = 1;
      v29 = 0;
      while ( v40 != -8 )
      {
        if ( !v29 && v40 == -16 )
          v29 = v41;
        v39 = v37 & (v42 + v39);
        v41 = (__int64 *)(v38 + 16LL * v39);
        v40 = *v41;
        if ( v26 == *v41 )
        {
LABEL_70:
          v29 = v41;
          goto LABEL_51;
        }
        ++v42;
      }
LABEL_59:
      if ( !v29 )
        v29 = v41;
      goto LABEL_51;
    }
LABEL_82:
    ++*(_DWORD *)(a1 + 928);
    BUG();
  }
  return result;
}
