// Function: sub_D2AB20
// Address: 0xd2ab20
//
_DWORD *__fastcall sub_D2AB20(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rdx
  _DWORD *result; // rax
  __int64 v10; // r15
  __int64 v11; // r12
  __int64 v12; // r8
  int v13; // esi
  unsigned int v14; // edx
  _QWORD *v15; // rax
  __int64 v16; // r9
  int v17; // r14d
  __int64 *v18; // r13
  char v19; // cl
  unsigned int v20; // esi
  unsigned int v21; // eax
  int v22; // edx
  unsigned int v23; // edi
  __int64 v24; // rax
  __int64 v25; // rdx
  int v26; // r11d
  _QWORD *v27; // r10
  __int64 v28; // [rsp+18h] [rbp-48h]
  _QWORD v29[8]; // [rsp+20h] [rbp-40h] BYREF

  v6 = *a3;
  v7 = a3[1];
  if ( *a3 != v7 )
  {
    do
    {
      v8 = *(_QWORD *)(v6 - 8);
      v6 -= 8;
      *(_QWORD *)(v8 + 16) = 0;
    }
    while ( v7 != v6 );
  }
  v29[0] = a1;
  v29[1] = a2;
  result = (_DWORD *)sub_D2A080(a3, (__int64)v29);
  v10 = *(unsigned int *)(a2 + 16);
  if ( (int)v10 > 0 )
  {
    v11 = 0;
    v28 = a2 + 56;
    while ( 1 )
    {
      v17 = v11;
      v18 = (__int64 *)(*(_QWORD *)(a2 + 8) + 8 * v11);
      v19 = *(_BYTE *)(a2 + 64) & 1;
      if ( v19 )
      {
        v12 = a2 + 72;
        v13 = 3;
      }
      else
      {
        v20 = *(_DWORD *)(a2 + 80);
        v12 = *(_QWORD *)(a2 + 72);
        if ( !v20 )
        {
          v21 = *(_DWORD *)(a2 + 64);
          ++*(_QWORD *)(a2 + 56);
          v29[0] = 0;
          v22 = (v21 >> 1) + 1;
LABEL_12:
          v23 = 3 * v20;
          goto LABEL_13;
        }
        v13 = v20 - 1;
      }
      v14 = v13 & (((unsigned int)*v18 >> 9) ^ ((unsigned int)*v18 >> 4));
      v15 = (_QWORD *)(v12 + 16LL * v14);
      v16 = *v15;
      if ( *v18 == *v15 )
      {
LABEL_7:
        ++v11;
        result = v15 + 1;
        *result = v17;
        if ( v11 == v10 )
          return result;
      }
      else
      {
        v26 = 1;
        v27 = 0;
        while ( v16 != -4096 )
        {
          if ( !v27 && v16 == -8192 )
            v27 = v15;
          v14 = v13 & (v26 + v14);
          v15 = (_QWORD *)(v12 + 16LL * v14);
          v16 = *v15;
          if ( *v18 == *v15 )
            goto LABEL_7;
          ++v26;
        }
        v23 = 12;
        v20 = 4;
        if ( !v27 )
          v27 = v15;
        v21 = *(_DWORD *)(a2 + 64);
        ++*(_QWORD *)(a2 + 56);
        v29[0] = v27;
        v22 = (v21 >> 1) + 1;
        if ( !v19 )
        {
          v20 = *(_DWORD *)(a2 + 80);
          goto LABEL_12;
        }
LABEL_13:
        if ( 4 * v22 >= v23 )
        {
          v20 *= 2;
LABEL_20:
          sub_D257F0(v28, v20);
          sub_D24B80(v28, v18, v29);
          v21 = *(_DWORD *)(a2 + 64);
          goto LABEL_15;
        }
        if ( v20 - *(_DWORD *)(a2 + 68) - v22 <= v20 >> 3 )
          goto LABEL_20;
LABEL_15:
        *(_DWORD *)(a2 + 64) = (2 * (v21 >> 1) + 2) | v21 & 1;
        v24 = v29[0];
        if ( *(_QWORD *)v29[0] != -4096 )
          --*(_DWORD *)(a2 + 68);
        v25 = *v18;
        result = (_DWORD *)(v24 + 8);
        ++v11;
        *result = 0;
        *((_QWORD *)result - 1) = v25;
        *result = v17;
        if ( v11 == v10 )
          return result;
      }
    }
  }
  return result;
}
