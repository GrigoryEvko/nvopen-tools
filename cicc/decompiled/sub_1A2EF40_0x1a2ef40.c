// Function: sub_1A2EF40
// Address: 0x1a2ef40
//
__int64 __fastcall sub_1A2EF40(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // rax
  __int64 v12; // r12
  int v13; // eax
  int v14; // edx
  __int64 v15; // rdi
  unsigned int v16; // eax
  __int64 *v17; // rcx
  __int64 v18; // rsi
  __int64 *v19; // rax
  __int64 *v20; // r14
  unsigned __int64 *v21; // r15
  __int64 *v22; // rbx
  _QWORD *v23; // rdi
  _QWORD *v24; // rbx
  __int64 v25; // rax
  double v26; // xmm4_8
  double v27; // xmm5_8
  _QWORD *v28; // rdx
  _QWORD *i; // r15
  __int64 v30; // rcx
  unsigned __int64 v31; // rax
  __int64 *v33; // rsi
  unsigned int v34; // edi
  __int64 *v35; // rcx
  int v36; // ecx
  int v37; // r8d
  __int64 v39; // [rsp+8h] [rbp-48h]
  __int64 v40; // [rsp+18h] [rbp-38h] BYREF
  char v41; // [rsp+20h] [rbp-30h] BYREF

  v10 = *(unsigned int *)(a1 + 248);
  if ( (_DWORD)v10 )
  {
    v39 = a1 + 208;
    while ( 1 )
    {
      v12 = *(_QWORD *)(*(_QWORD *)(a1 + 240) + 8 * v10 - 8);
      v13 = *(_DWORD *)(a1 + 232);
      if ( v13 )
      {
        v14 = v13 - 1;
        v15 = *(_QWORD *)(a1 + 216);
        v16 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v17 = (__int64 *)(v15 + 8LL * v16);
        v18 = *v17;
        if ( v12 == *v17 )
        {
LABEL_5:
          *v17 = -16;
          --*(_DWORD *)(a1 + 224);
          ++*(_DWORD *)(a1 + 228);
        }
        else
        {
          v36 = 1;
          while ( v18 != -8 )
          {
            v37 = v36 + 1;
            v16 = v14 & (v36 + v16);
            v17 = (__int64 *)(v15 + 8LL * v16);
            v18 = *v17;
            if ( v12 == *v17 )
              goto LABEL_5;
            v36 = v37;
          }
        }
      }
      --*(_DWORD *)(a1 + 248);
      if ( *(_BYTE *)(v12 + 16) != 53 )
        goto LABEL_18;
      v19 = *(__int64 **)(a2 + 8);
      if ( *(__int64 **)(a2 + 16) != v19 )
        break;
      v33 = &v19[*(unsigned int *)(a2 + 28)];
      v34 = *(_DWORD *)(a2 + 28);
      if ( v19 == v33 )
      {
LABEL_45:
        if ( v34 >= *(_DWORD *)(a2 + 24) )
          break;
        *(_DWORD *)(a2 + 28) = v34 + 1;
        *v33 = v12;
        ++*(_QWORD *)a2;
      }
      else
      {
        v35 = 0;
        while ( v12 != *v19 )
        {
          if ( *v19 == -2 )
            v35 = v19;
          if ( v33 == ++v19 )
          {
            if ( !v35 )
              goto LABEL_45;
            *v35 = v12;
            --*(_DWORD *)(a2 + 32);
            ++*(_QWORD *)a2;
            break;
          }
        }
      }
LABEL_9:
      v20 = &v40;
      sub_1AEA030(&v40);
      if ( (v40 & 4) != 0 )
      {
        v20 = *(__int64 **)(v40 & 0xFFFFFFFFFFFFFFF8LL);
        v21 = (unsigned __int64 *)(v40 & 0xFFFFFFFFFFFFFFF8LL);
        v22 = &v20[*(unsigned int *)((v40 & 0xFFFFFFFFFFFFFFF8LL) + 8)];
        if ( v20 == v22 )
          goto LABEL_14;
      }
      else
      {
        v22 = (__int64 *)&v41;
        if ( (v40 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          goto LABEL_18;
      }
      do
      {
        v23 = (_QWORD *)*v20++;
        sub_15F20C0(v23);
      }
      while ( v22 != v20 );
      if ( (v40 & 4) != 0 )
      {
        v21 = (unsigned __int64 *)(v40 & 0xFFFFFFFFFFFFFFF8LL);
LABEL_14:
        if ( v21 )
        {
          if ( (unsigned __int64 *)*v21 != v21 + 2 )
            _libc_free(*v21);
          j_j___libc_free_0(v21, 48);
        }
      }
LABEL_18:
      v24 = (_QWORD *)v12;
      v25 = sub_1599EF0(*(__int64 ***)v12);
      sub_164D160(v12, v25, a3, a4, a5, a6, v26, v27, a9, a10);
      if ( (*(_BYTE *)(v12 + 23) & 0x40) != 0 )
      {
        v28 = *(_QWORD **)(v12 - 8);
        v24 = &v28[3 * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF)];
      }
      else
      {
        v28 = (_QWORD *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF));
      }
      for ( i = v28; v24 != i; i += 3 )
      {
        while ( 1 )
        {
          if ( *(_BYTE *)(*i + 16LL) > 0x17u )
          {
            v40 = *i;
            v30 = i[1];
            v31 = i[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v31 = v30;
            if ( v30 )
              *(_QWORD *)(v30 + 16) = *(_QWORD *)(v30 + 16) & 3LL | v31;
            *i = 0;
            if ( (unsigned __int8)sub_1AE9990(v40, 0) )
              break;
          }
          i += 3;
          if ( v24 == i )
            goto LABEL_28;
        }
        sub_1A2EDE0(v39, &v40);
      }
LABEL_28:
      sub_15F20C0((_QWORD *)v12);
      v10 = *(unsigned int *)(a1 + 248);
      if ( !(_DWORD)v10 )
        return 1;
    }
    sub_16CCBA0(a2, v12);
    goto LABEL_9;
  }
  return 0;
}
