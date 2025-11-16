// Function: sub_257B840
// Address: 0x257b840
//
__int64 __fastcall sub_257B840(_QWORD *a1)
{
  unsigned __int64 *v1; // rbx
  __int64 result; // rax
  unsigned __int64 *v3; // r14
  _QWORD *v5; // rdi
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 *v11; // r15
  __int64 v12; // r12
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r12
  __int64 v16; // rsi
  __int64 (__fastcall *v17)(__int64, __int64); // rax
  __int64 v18; // r12
  unsigned int v19; // esi
  __int64 v20; // r9
  _QWORD *v21; // r11
  int v22; // r15d
  unsigned int v23; // edx
  _QWORD *v24; // r8
  __int64 v25; // rdi
  int v26; // eax
  unsigned __int64 *v27; // rax
  unsigned __int64 v28; // r15
  __int64 v29; // [rsp-8h] [rbp-58h]
  __int64 *v30; // [rsp+8h] [rbp-48h]
  unsigned __int64 v31; // [rsp+10h] [rbp-40h] BYREF
  unsigned __int64 *v32; // [rsp+18h] [rbp-38h] BYREF

  v1 = *(unsigned __int64 **)(*a1 + 168LL);
  result = *(unsigned int *)(*a1 + 176LL);
  v3 = &v1[result];
  if ( v3 != v1 )
  {
    while ( 1 )
    {
      v11 = (__int64 *)a1[1];
      v31 = *v1;
      v12 = *v11;
      v13 = sub_250D2C0(v31, 0);
      v15 = sub_257B470(v12, v13, v14, v11[1], 1, 0, 1);
      if ( v15 )
      {
        v16 = v11[2];
        v17 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v15 + 112LL);
        if ( v17 == sub_254A530 )
        {
          if ( *(_BYTE *)(v15 + 97) && !(unsigned __int8)sub_B19060(v15 + 104, v16, v29, (__int64)sub_254A530) )
          {
LABEL_24:
            result = *(_QWORD *)(*(_QWORD *)(v15 + 88) + 24LL);
            if ( (bool (__fastcall *)(__int64))result != sub_2534ED0 )
              result = ((__int64 (__fastcall *)(__int64))result)(v15 + 88);
            goto LABEL_5;
          }
        }
        else if ( !(unsigned __int8)v17(v15, v16) )
        {
          goto LABEL_24;
        }
      }
      v18 = a1[2];
      result = *(unsigned int *)(v18 + 16);
      if ( !(_DWORD)result )
      {
        v5 = *(_QWORD **)(v18 + 32);
        v6 = (__int64)&v5[*(unsigned int *)(v18 + 40)];
        result = (__int64)sub_2538080(v5, v6, (__int64 *)&v31);
        if ( v6 == result )
          result = sub_25718D0(v18, v31, v7, v8, v9, v10);
        goto LABEL_5;
      }
      v19 = *(_DWORD *)(v18 + 24);
      if ( !v19 )
        break;
      v20 = *(_QWORD *)(v18 + 8);
      v21 = 0;
      v22 = 1;
      v23 = (v19 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
      v24 = (_QWORD *)(v20 + 8LL * v23);
      v25 = *v24;
      if ( v31 == *v24 )
      {
LABEL_5:
        if ( v3 == ++v1 )
          return result;
      }
      else
      {
        while ( v25 != -4096 )
        {
          if ( v25 != -8192 || v21 )
            v24 = v21;
          v23 = (v19 - 1) & (v22 + v23);
          v30 = (__int64 *)(v20 + 8LL * v23);
          v25 = *v30;
          if ( v31 == *v30 )
            goto LABEL_5;
          ++v22;
          v21 = v24;
          v24 = (_QWORD *)(v20 + 8LL * v23);
        }
        if ( !v21 )
          v21 = v24;
        v26 = result + 1;
        v32 = v21;
        ++*(_QWORD *)v18;
        if ( 4 * v26 >= 3 * v19 )
          goto LABEL_29;
        if ( v19 - *(_DWORD *)(v18 + 20) - v26 <= v19 >> 3 )
          goto LABEL_30;
LABEL_17:
        *(_DWORD *)(v18 + 16) = v26;
        v27 = v32;
        if ( *v32 != -4096 )
          --*(_DWORD *)(v18 + 20);
        *v27 = v31;
        result = *(unsigned int *)(v18 + 40);
        v28 = v31;
        if ( result + 1 > (unsigned __int64)*(unsigned int *)(v18 + 44) )
        {
          sub_C8D5F0(v18 + 32, (const void *)(v18 + 48), result + 1, 8u, (__int64)v24, v20);
          result = *(unsigned int *)(v18 + 40);
        }
        ++v1;
        *(_QWORD *)(*(_QWORD *)(v18 + 32) + 8 * result) = v28;
        ++*(_DWORD *)(v18 + 40);
        if ( v3 == v1 )
          return result;
      }
    }
    v32 = 0;
    ++*(_QWORD *)v18;
LABEL_29:
    v19 *= 2;
LABEL_30:
    sub_A35F10(v18, v19);
    sub_A2AFD0(v18, (__int64 *)&v31, &v32);
    v26 = *(_DWORD *)(v18 + 16) + 1;
    goto LABEL_17;
  }
  return result;
}
