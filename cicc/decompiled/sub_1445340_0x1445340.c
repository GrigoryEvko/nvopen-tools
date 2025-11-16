// Function: sub_1445340
// Address: 0x1445340
//
signed __int64 __fastcall sub_1445340(__int64 a1)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // r15
  __int64 v4; // r13
  __int64 v5; // rax
  int v6; // ecx
  signed __int64 result; // rax
  unsigned __int64 v8; // r13
  unsigned __int64 v9; // rsi
  _QWORD *v10; // r13
  _QWORD *v11; // rax
  char v12; // dl
  int v13; // esi
  __int64 v14; // r8
  unsigned int v15; // r14d
  __int64 v16; // rdi
  int v17; // eax
  _QWORD *v18; // rcx
  unsigned int v19; // edi
  _QWORD *v20; // rsi
  __int64 v21; // r14
  __int64 v22; // r13
  __int64 v23; // rdi
  unsigned int i; // r14d
  int v25; // eax
  __int64 v26; // [rsp+0h] [rbp-80h]
  unsigned __int64 v27; // [rsp+0h] [rbp-80h]
  unsigned __int64 v28; // [rsp+8h] [rbp-78h]
  unsigned __int64 v29; // [rsp+10h] [rbp-70h]
  unsigned int v30; // [rsp+1Ch] [rbp-64h]
  _QWORD *v31; // [rsp+20h] [rbp-60h] BYREF
  char v32; // [rsp+40h] [rbp-40h]

  v2 = *(_QWORD *)(a1 + 112);
  do
  {
    v3 = *(_QWORD *)(v2 - 40);
    v29 = v3 & 0xFFFFFFFFFFFFFFF9LL;
    if ( !*(_BYTE *)(v2 - 8) )
    {
      v21 = (*(_QWORD *)v3 >> 1) & 2LL;
      v22 = sub_157EBA0(*(_QWORD *)v3 & 0xFFFFFFFFFFFFFFF8LL);
      v28 = v29 | v21;
      v27 = v29 | v21;
      if ( (v21 & 2) != 0 )
      {
        if ( *(_QWORD *)((v3 & 0xFFFFFFFFFFFFFFF8LL | v21 & 0xFFFFFFFFFFFFFFF8LL) + 0x20) == *(_QWORD *)(*(_QWORD *)((v3 & 0xFFFFFFFFFFFFFFF8LL | v21 & 0xFFFFFFFFFFFFFFF8LL) + 8) + 32LL) )
          v27 = v27 & 0xFFFFFFFFFFFFFFF9LL | 4;
        i = 0;
      }
      else
      {
        v23 = v22;
        for ( i = 0; ; ++i )
        {
          v25 = 0;
          if ( v23 )
            v25 = sub_15F4D60(v23);
          if ( v25 == i || *(_QWORD *)(*(_QWORD *)((v28 & 0xFFFFFFFFFFFFFFF8LL) + 8) + 32LL) != sub_15F4DF0(v22, i) )
            break;
          v23 = sub_157EBA0(*(_QWORD *)v3 & 0xFFFFFFFFFFFFFFF8LL);
        }
      }
      *(_BYTE *)(v2 - 8) = 1;
      *(_QWORD *)(v2 - 24) = v22;
      *(_QWORD *)(v2 - 32) = v27;
      *(_DWORD *)(v2 - 16) = i;
    }
    while ( 1 )
    {
      v4 = v3 & 0xFFFFFFFFFFFFFFF9LL;
      if ( (*(_QWORD *)v3 & 4) != 0 )
        v4 = v29 | 4;
      v5 = sub_157EBA0(*(_QWORD *)v3 & 0xFFFFFFFFFFFFFFF8LL);
      v6 = 0;
      if ( v5 )
        v6 = sub_15F4D60(v5);
      result = *(_QWORD *)(v2 - 32);
      if ( ((result >> 1) & 3) == 0 )
        break;
      if ( ((result >> 1) & 3) == ((v4 >> 1) & 3) )
        goto LABEL_32;
      v8 = result & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v2 - 32) = *(_QWORD *)(v2 - 32) & 0xFFFFFFFFFFFFFFF9LL | 4;
      v9 = *(_QWORD *)((result & 0xFFFFFFFFFFFFFFF8LL) + 32);
LABEL_10:
      v10 = sub_1444E60(*(_QWORD **)(v8 + 8), v9);
      v11 = *(_QWORD **)(a1 + 8);
      if ( *(_QWORD **)(a1 + 16) != v11 )
        goto LABEL_11;
      v18 = &v11[*(unsigned int *)(a1 + 28)];
      v19 = *(_DWORD *)(a1 + 28);
      if ( v11 == v18 )
      {
LABEL_30:
        if ( v19 < *(_DWORD *)(a1 + 24) )
        {
          *(_DWORD *)(a1 + 28) = v19 + 1;
          *v18 = v10;
          ++*(_QWORD *)a1;
LABEL_12:
          v31 = v10;
          v32 = 0;
          return sub_14452F0((__int64 *)(a1 + 104), (__int64)&v31);
        }
LABEL_11:
        sub_16CCBA0(a1, v10);
        if ( v12 )
          goto LABEL_12;
      }
      else
      {
        v20 = 0;
        while ( v10 != (_QWORD *)*v11 )
        {
          if ( *v11 == -2 )
          {
            v20 = v11;
            if ( v11 + 1 == v18 )
              goto LABEL_27;
            ++v11;
          }
          else if ( v18 == ++v11 )
          {
            if ( !v20 )
              goto LABEL_30;
LABEL_27:
            *v20 = v10;
            --*(_DWORD *)(a1 + 32);
            ++*(_QWORD *)a1;
            goto LABEL_12;
          }
        }
      }
    }
    v30 = *(_DWORD *)(v2 - 16);
    if ( v30 != v6 )
    {
      v13 = *(_DWORD *)(v2 - 16);
      v8 = result & 0xFFFFFFFFFFFFFFF8LL;
      v26 = *(_QWORD *)(v2 - 24);
      while ( 1 )
      {
        v15 = v13 + 1;
        *(_DWORD *)(v2 - 16) = v13 + 1;
        v16 = sub_157EBA0(*(_QWORD *)(result & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL);
        v17 = 0;
        if ( v16 )
        {
          v17 = sub_15F4D60(v16);
          v15 = *(_DWORD *)(v2 - 16);
        }
        if ( v17 == v15 )
          break;
        v14 = sub_15F4DF0(*(_QWORD *)(v2 - 24), v15);
        result = *(_QWORD *)(v2 - 32);
        if ( *(_QWORD *)(*(_QWORD *)((result & 0xFFFFFFFFFFFFFFF8LL) + 8) + 32LL) != v14 )
          break;
        v13 = *(_DWORD *)(v2 - 16);
      }
      v9 = sub_15F4DF0(v26, v30);
      goto LABEL_10;
    }
LABEL_32:
    *(_QWORD *)(a1 + 112) -= 40LL;
    v2 = *(_QWORD *)(a1 + 112);
  }
  while ( v2 != *(_QWORD *)(a1 + 104) );
  return result;
}
