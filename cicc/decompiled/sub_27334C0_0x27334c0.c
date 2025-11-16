// Function: sub_27334C0
// Address: 0x27334c0
//
__int64 __fastcall sub_27334C0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v8; // r13
  __int64 *v9; // rdi
  int v11; // edx
  __int64 v12; // rcx
  __int64 v13; // r9
  __int64 v14; // r15
  unsigned __int64 *v15; // r8
  int v16; // eax
  __int64 v17; // rdx
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rax
  int v21; // [rsp+0h] [rbp-100h]
  unsigned __int64 v22; // [rsp+10h] [rbp-F0h] BYREF
  int v23; // [rsp+18h] [rbp-E8h] BYREF
  unsigned __int64 v24[2]; // [rsp+20h] [rbp-E0h] BYREF
  _QWORD v25[2]; // [rsp+30h] [rbp-D0h] BYREF
  char v26; // [rsp+40h] [rbp-C0h]
  __int64 v27; // [rsp+B0h] [rbp-50h]
  __int64 v28; // [rsp+B8h] [rbp-48h]
  int v29; // [rsp+C0h] [rbp-40h]

  result = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a5 + 8) + 8LL) - 17;
  if ( (unsigned int)result > 1 )
  {
    v8 = a4;
    v9 = *(__int64 **)a1;
    if ( *(_BYTE *)a3 == 85
      && (v20 = *(_QWORD *)(a3 - 32)) != 0
      && !*(_BYTE *)v20
      && *(_QWORD *)(v20 + 24) == *(_QWORD *)(a3 + 80)
      && (*(_BYTE *)(v20 + 33) & 0x20) != 0 )
    {
      result = sub_DFB0A0((__int64)v9);
    }
    else
    {
      result = sub_DFB040(v9);
    }
    if ( v11 )
    {
      if ( v11 <= 0 )
        return result;
    }
    else if ( result <= 1 )
    {
      return result;
    }
    v21 = result;
    v22 = a5 & 0xFFFFFFFFFFFFFFFBLL;
    v23 = 0;
    sub_2733220((__int64)v24, a2, (__int64 *)&v22, &v23);
    v14 = v25[0];
    v15 = v24;
    v16 = v21;
    if ( v26 )
    {
      v27 = a5;
      v24[0] = (unsigned __int64)v25;
      v24[1] = 0x800000000LL;
      v28 = 0;
      v29 = 0;
      sub_2731550((unsigned __int64 *)(a1 + 64), (__int64)v24, (__int64)v25, v12, (__int64)v24, v13);
      v16 = v21;
      if ( (_QWORD *)v24[0] != v25 )
      {
        _libc_free(v24[0]);
        v16 = v21;
      }
      v17 = 1022611261 * (unsigned int)((__int64)(*(_QWORD *)(a1 + 72) - *(_QWORD *)(a1 + 64)) >> 3) - 1;
      *(_DWORD *)(v14 + 8) = v17;
    }
    else
    {
      v17 = *(unsigned int *)(v25[0] + 8LL);
    }
    v18 = *(_QWORD *)(a1 + 64) + 168 * v17;
    *(_DWORD *)(v18 + 160) += v16;
    v19 = *(unsigned int *)(v18 + 8);
    if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(v18 + 12) )
    {
      sub_C8D5F0(v18, (const void *)(v18 + 16), v19 + 1, 0x10u, (__int64)v15, v13);
      v19 = *(unsigned int *)(v18 + 8);
    }
    result = *(_QWORD *)v18 + 16 * v19;
    *(_QWORD *)result = a3;
    *(_QWORD *)(result + 8) = v8;
    ++*(_DWORD *)(v18 + 8);
  }
  return result;
}
