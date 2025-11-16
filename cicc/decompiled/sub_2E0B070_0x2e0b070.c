// Function: sub_2E0B070
// Address: 0x2e0b070
//
__int64 __fastcall sub_2E0B070(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, __int64 a6)
{
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // r9
  __int64 v13; // r10
  __int64 result; // rax
  __int64 v15; // rbx
  __int64 v16; // r13
  char v17; // dl
  _QWORD *v18; // rax
  __int64 v19; // rcx
  unsigned __int64 v20; // rsi
  char v21; // dl
  unsigned __int64 i; // rax
  __int64 j; // rdi
  __int16 v24; // cx
  __int64 v25; // rdi
  __int64 v26; // r8
  unsigned int v27; // esi
  __int64 *v28; // rcx
  __int64 v29; // r11
  unsigned __int64 v30; // r15
  int v31; // ecx
  __int64 v33; // [rsp+8h] [rbp-48h]
  __int64 v35; // [rsp+10h] [rbp-40h]
  int v36; // [rsp+10h] [rbp-40h]
  __int64 v37; // [rsp+18h] [rbp-38h]

  v37 = sub_2EBF1E0(a5, *(unsigned int *)(a1 + 112), a3, a4, a5, a6);
  v10 = v9;
  v11 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*a5 + 16LL) + 200LL))(*(_QWORD *)(*a5 + 16LL));
  v12 = a6;
  v13 = v11;
  result = *(unsigned int *)(a1 + 112);
  if ( (int)result < 0 )
  {
    result = a5[7] + 16 * (result & 0x7FFFFFFF);
    v15 = *(_QWORD *)(result + 8);
  }
  else
  {
    v15 = *(_QWORD *)(a5[38] + 8 * result);
  }
  if ( v15 )
  {
    if ( (*(_BYTE *)(v15 + 3) & 0x10) != 0 || (v15 = *(_QWORD *)(v15 + 32)) != 0 && (*(_BYTE *)(v15 + 3) & 0x10) != 0 )
    {
      v16 = a4 & v10;
      while ( 1 )
      {
        v17 = *(_BYTE *)(v15 + 4);
        if ( (v17 & 1) != 0 )
        {
          v18 = (_QWORD *)(*(_QWORD *)(v13 + 272) + 16LL * ((*(_DWORD *)v15 >> 8) & 0xFFF));
          v19 = v18[1];
          result = a3 & v37 & ~*v18;
          if ( result | v16 & ~v19 )
            break;
        }
LABEL_7:
        v15 = *(_QWORD *)(v15 + 32);
        if ( !v15 || (*(_BYTE *)(v15 + 3) & 0x10) == 0 )
          return result;
      }
      v20 = *(_QWORD *)(v15 + 16);
      v21 = v17 & 4;
      for ( i = v20; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
        ;
      for ( ; (*(_BYTE *)(v20 + 44) & 8) != 0; v20 = *(_QWORD *)(v20 + 8) )
        ;
      for ( j = *(_QWORD *)(v20 + 8); j != i; i = *(_QWORD *)(i + 8) )
      {
        v24 = *(_WORD *)(i + 68);
        if ( (unsigned __int16)(v24 - 14) > 4u && v24 != 24 )
          break;
      }
      v25 = *(unsigned int *)(v12 + 144);
      v26 = *(_QWORD *)(v12 + 128);
      if ( (_DWORD)v25 )
      {
        v27 = (v25 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
        v28 = (__int64 *)(v26 + 16LL * v27);
        v29 = *v28;
        if ( i == *v28 )
        {
LABEL_21:
          v30 = (v21 == 0 ? 4LL : 2LL) | v28[1] & 0xFFFFFFFFFFFFFFF8LL;
          result = *(unsigned int *)(a2 + 8);
          if ( result + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
          {
            v33 = v12;
            v35 = v13;
            sub_C8D5F0(a2, (const void *)(a2 + 16), result + 1, 8u, v26, v12);
            result = *(unsigned int *)(a2 + 8);
            v12 = v33;
            v13 = v35;
          }
          *(_QWORD *)(*(_QWORD *)a2 + 8 * result) = v30;
          ++*(_DWORD *)(a2 + 8);
          goto LABEL_7;
        }
        v31 = 1;
        while ( v29 != -4096 )
        {
          v27 = (v25 - 1) & (v31 + v27);
          v36 = v31 + 1;
          v28 = (__int64 *)(v26 + 16LL * v27);
          v29 = *v28;
          if ( i == *v28 )
            goto LABEL_21;
          v31 = v36;
        }
      }
      v28 = (__int64 *)(v26 + 16 * v25);
      goto LABEL_21;
    }
  }
  return result;
}
