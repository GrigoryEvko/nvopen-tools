// Function: sub_25FFC80
// Address: 0x25ffc80
//
char __fastcall sub_25FFC80(__int64 a1, unsigned int **a2)
{
  unsigned int *v4; // rbx
  unsigned int v5; // eax
  unsigned int v6; // r8d
  __int64 v7; // r10
  __int64 v8; // r9
  int v9; // edx
  unsigned int v10; // r11d
  unsigned int v11; // edi
  int *v12; // rcx
  int v13; // esi
  int v14; // ecx
  int v15; // r15d
  __int64 v16; // rdx
  unsigned __int8 *v17; // rdi
  __int64 v18; // r13
  __int64 v19; // rbx
  __int64 v20; // r12
  char result; // al
  __int64 v22; // r15
  unsigned __int8 v23; // al
  unsigned __int64 v24; // r9
  __int64 v25; // r8
  __int64 v26; // rax
  unsigned __int64 v27; // r14
  unsigned __int64 *v28; // rdx
  unsigned __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  unsigned int v33; // [rsp+4h] [rbp-3Ch]
  unsigned __int64 v34; // [rsp+8h] [rbp-38h]

  v4 = *a2;
  v5 = **a2;
  v6 = v5 + (*a2)[1] - 1;
  if ( v5 <= v6 )
  {
    v7 = *(_QWORD *)(a1 + 16);
    v8 = *(unsigned int *)(a1 + 32);
    v9 = 37 * v5;
    v10 = v8 - 1;
    do
    {
      if ( (_DWORD)v8 )
      {
        v11 = v9 & v10;
        v12 = (int *)(v7 + 4LL * (v9 & v10));
        v13 = *v12;
        if ( v5 == *v12 )
        {
LABEL_3:
          if ( (int *)(v7 + 4 * v8) != v12 )
            return 0;
        }
        else
        {
          v14 = 1;
          while ( v13 != -1 )
          {
            v15 = v14 + 1;
            v11 = v10 & (v14 + v11);
            v12 = (int *)(v7 + 4LL * v11);
            v13 = *v12;
            if ( *v12 == v5 )
              goto LABEL_3;
            v14 = v15;
          }
        }
      }
      ++v5;
      v9 += 37;
    }
    while ( v5 <= v6 );
  }
  v16 = *((_QWORD *)v4 + 2);
  v17 = *(unsigned __int8 **)(v16 + 16);
  if ( (unsigned int)*v17 - 30 <= 0xA )
  {
    v18 = *(_QWORD *)(v16 + 8);
    v19 = *((_QWORD *)v4 + 1);
    if ( v19 != v18 )
      goto LABEL_13;
    return 1;
  }
  v22 = sub_B46B10((__int64)v17, 0);
  if ( *(_QWORD *)(*(_QWORD *)(*((_QWORD *)*a2 + 2) + 8LL) + 16LL) != v22 )
  {
    v34 = *(_QWORD *)(*((_QWORD *)*a2 + 1) + 160LL);
    v23 = sub_25FE580(a1 + 408, v22);
    v24 = v34;
    *(_QWORD *)(a1 + 392) += 168LL;
    v25 = v23;
    v26 = *(_QWORD *)(a1 + 312);
    v27 = (v26 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( *(_QWORD *)(a1 + 320) >= v27 + 168 && v26 )
    {
      *(_QWORD *)(a1 + 312) = v27 + 168;
    }
    else
    {
      v33 = v25;
      v32 = sub_9D1E70(a1 + 312, 168, 168, 3);
      v25 = v33;
      v24 = v34;
      v27 = v32;
    }
    sub_22AF450(v27, v22, v25, v24, v25, v24);
    v28 = *(unsigned __int64 **)(*((_QWORD *)*a2 + 2) + 8LL);
    v29 = *v28;
    v30 = *(_QWORD *)v27 & 7LL;
    *(_QWORD *)(v27 + 8) = v28;
    v29 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v27 = v29 | v30;
    *(_QWORD *)(v29 + 8) = v27;
    *v28 = *v28 & 7 | v27;
  }
  v31 = *((_QWORD *)v4 + 2);
  v19 = *((_QWORD *)v4 + 1);
  v18 = *(_QWORD *)(v31 + 8);
  if ( v19 == v18 )
    return 1;
LABEL_13:
  v20 = a1 + 408;
  while ( (unsigned __int8)sub_25F86A0(v19) )
  {
    result = sub_25FE580(v20, *(_QWORD *)(v19 + 16));
    if ( !result )
      break;
    v19 = *(_QWORD *)(v19 + 8);
    if ( v19 == v18 )
      return result;
  }
  return v18 == v19;
}
