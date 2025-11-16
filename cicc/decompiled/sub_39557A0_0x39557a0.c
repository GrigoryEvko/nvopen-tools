// Function: sub_39557A0
// Address: 0x39557a0
//
__int64 __fastcall sub_39557A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v4; // esi
  __int64 v5; // rcx
  unsigned int v7; // r9d
  __int64 v8; // r8
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // rdi
  unsigned int v13; // eax
  char v14; // cl
  __int64 v15; // rax
  __int64 v16; // r10
  __int64 v17; // r13
  int i; // r11d
  int v19; // r13d
  __int64 *v20; // r10
  int v21; // eax
  int v22; // edx
  __int64 *v23; // r11
  unsigned int v24; // eax
  __int64 v25; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v26[5]; // [rsp+18h] [rbp-28h] BYREF

  result = 0;
  v25 = a2;
  v4 = *(_DWORD *)(a1 + 80);
  if ( v4 )
  {
    v5 = v25;
    v7 = v4 - 1;
    v8 = *(_QWORD *)(a1 + 64);
    v10 = (v4 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
    v11 = (__int64 *)(v8 + 16LL * v10);
    v12 = *v11;
    if ( v25 == *v11 )
    {
      v13 = *((_DWORD *)v11 + 2);
      v14 = v13 & 0x3F;
      v15 = 8LL * (v13 >> 6);
      return (*(_QWORD *)(*(_QWORD *)(a3 + 24) + v15) >> v14) & 1LL;
    }
    v16 = *v11;
    LODWORD(v17) = (v4 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
    for ( i = 1; ; ++i )
    {
      if ( v16 == -8 )
        return 0;
      v17 = v7 & ((_DWORD)v17 + i);
      v16 = *(_QWORD *)(v8 + 16 * v17);
      if ( v25 == v16 )
        break;
    }
    v19 = 1;
    v20 = 0;
    while ( v12 != -8 )
    {
      if ( v20 || v12 != -16 )
        v11 = v20;
      v10 = v7 & (v19 + v10);
      v23 = (__int64 *)(v8 + 16LL * v10);
      v12 = *v23;
      if ( v25 == *v23 )
      {
        v24 = *((_DWORD *)v23 + 2);
        v14 = v24 & 0x3F;
        v15 = 8LL * (v24 >> 6);
        return (*(_QWORD *)(*(_QWORD *)(a3 + 24) + v15) >> v14) & 1LL;
      }
      ++v19;
      v20 = v11;
      v11 = (__int64 *)(v8 + 16LL * v10);
    }
    if ( !v20 )
      v20 = v11;
    v21 = *(_DWORD *)(a1 + 72);
    ++*(_QWORD *)(a1 + 56);
    v22 = v21 + 1;
    if ( 4 * (v21 + 1) >= 3 * v4 )
    {
      v4 *= 2;
    }
    else if ( v4 - *(_DWORD *)(a1 + 76) - v22 > v4 >> 3 )
    {
LABEL_16:
      *(_DWORD *)(a1 + 72) = v22;
      if ( *v20 != -8 )
        --*(_DWORD *)(a1 + 76);
      *v20 = v5;
      v15 = 0;
      v14 = 0;
      *((_DWORD *)v20 + 2) = 0;
      return (*(_QWORD *)(*(_QWORD *)(a3 + 24) + v15) >> v14) & 1LL;
    }
    sub_1BFE340(a1 + 56, v4);
    sub_1BFD9C0(a1 + 56, &v25, v26);
    v20 = (__int64 *)v26[0];
    v5 = v25;
    v22 = *(_DWORD *)(a1 + 72) + 1;
    goto LABEL_16;
  }
  return result;
}
