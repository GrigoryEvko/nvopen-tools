// Function: sub_D3E0C0
// Address: 0xd3e0c0
//
__int64 __fastcall sub_D3E0C0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        int a7,
        int a8,
        __int64 a9,
        char a10)
{
  __int64 v13; // rax
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // r14
  __int64 v17; // rax
  int v18; // edx
  __int64 result; // rax
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v22; // r11
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // r9
  __int64 v26; // rsi
  __int64 v27; // rdi
  int v28; // r12d
  __int64 v29; // rdx
  __int64 v30; // [rsp+10h] [rbp-60h]
  __int64 v31; // [rsp+18h] [rbp-58h]
  __int64 v32; // [rsp+18h] [rbp-58h]
  __int64 v35; // [rsp+28h] [rbp-48h]
  __int64 v36; // [rsp+28h] [rbp-48h]
  __int64 v37; // [rsp+28h] [rbp-48h]
  unsigned __int64 v38[7]; // [rsp+38h] [rbp-38h] BYREF

  v13 = sub_DEF9D0(a9, a2);
  v14 = sub_D3B9E0(a2, a4, a5, v13, *(_QWORD *)(a9 + 112), *(_QWORD *)(a1 + 280) + 360LL);
  v16 = v15;
  v17 = *(unsigned int *)(a1 + 16);
  if ( *(_DWORD *)(a1 + 20) <= (unsigned int)v17 )
  {
    v32 = v14;
    v20 = a1 + 24;
    v21 = sub_C8D7D0(a1 + 8, a1 + 24, 0, 0x48u, v38, v14);
    v22 = a1 + 8;
    v23 = v21;
    v24 = v21 + 72LL * *(unsigned int *)(a1 + 16);
    if ( v24 )
    {
      *(_QWORD *)v24 = 6;
      v25 = v32;
      *(_QWORD *)(v24 + 8) = 0;
      if ( a3 )
      {
        *(_QWORD *)(v24 + 16) = a3;
        if ( a3 != -8192 && a3 != -4096 )
        {
          v30 = v23;
          v36 = v24;
          sub_BD73F0(v24);
          v25 = v32;
          v23 = v30;
          v22 = a1 + 8;
          v24 = v36;
        }
      }
      else
      {
        *(_QWORD *)(v24 + 16) = 0;
      }
      *(_QWORD *)(v24 + 24) = v25;
      *(_QWORD *)(v24 + 32) = v16;
      *(_BYTE *)(v24 + 40) = a6;
      *(_QWORD *)(v24 + 56) = a4;
      *(_DWORD *)(v24 + 44) = a7;
      *(_DWORD *)(v24 + 48) = a8;
      *(_BYTE *)(v24 + 64) = a10;
    }
    v26 = v23;
    v37 = v23;
    result = sub_D38870(v22, v23);
    v27 = *(_QWORD *)(a1 + 8);
    v28 = v38[0];
    v29 = v37;
    if ( v20 != v27 )
    {
      result = _libc_free(v27, v26);
      v29 = v37;
    }
    *(_DWORD *)(a1 + 20) = v28;
    ++*(_DWORD *)(a1 + 16);
    *(_QWORD *)(a1 + 8) = v29;
  }
  else
  {
    v18 = *(_DWORD *)(a1 + 16);
    result = *(_QWORD *)(a1 + 8) + 72 * v17;
    if ( result )
    {
      *(_QWORD *)result = 6;
      *(_QWORD *)(result + 8) = 0;
      if ( a3 )
      {
        *(_QWORD *)(result + 16) = a3;
        if ( a3 != -8192 && a3 != -4096 )
        {
          v31 = v14;
          v35 = result;
          sub_BD73F0(result);
          v14 = v31;
          result = v35;
        }
      }
      else
      {
        *(_QWORD *)(result + 16) = 0;
      }
      *(_QWORD *)(result + 24) = v14;
      *(_QWORD *)(result + 32) = v16;
      *(_BYTE *)(result + 40) = a6;
      *(_DWORD *)(result + 48) = a8;
      *(_DWORD *)(result + 44) = a7;
      *(_QWORD *)(result + 56) = a4;
      *(_BYTE *)(result + 64) = a10;
      v18 = *(_DWORD *)(a1 + 16);
    }
    *(_DWORD *)(a1 + 16) = v18 + 1;
  }
  return result;
}
