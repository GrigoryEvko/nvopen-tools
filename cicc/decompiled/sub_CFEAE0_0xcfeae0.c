// Function: sub_CFEAE0
// Address: 0xcfeae0
//
void __fastcall sub_CFEAE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v8; // rbx
  unsigned __int64 v9; // rcx
  char *v10; // r15
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rsi
  int v13; // edx
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdi
  char *v19; // r15
  _QWORD v20[2]; // [rsp-58h] [rbp-58h] BYREF
  __int64 v21; // [rsp-48h] [rbp-48h]
  int v22; // [rsp-40h] [rbp-40h]

  if ( *(_BYTE *)(a1 + 192) )
  {
    v20[0] = 4;
    v20[1] = 0;
    v21 = a2;
    if ( a2 != 0 && a2 != -4096 && a2 != -8192 )
      sub_BD73F0((__int64)v20);
    v8 = *(unsigned int *)(a1 + 24);
    v9 = *(unsigned int *)(a1 + 28);
    v22 = -1;
    v10 = (char *)v20;
    v11 = *(_QWORD *)(a1 + 16);
    v12 = v8 + 1;
    v13 = v8;
    if ( v8 + 1 > v9 )
    {
      v18 = a1 + 16;
      if ( v11 > (unsigned __int64)v20 || (unsigned __int64)v20 >= v11 + 32 * v8 )
      {
        sub_CFC2E0(v18, v12, v8, v9, a5, (__int64)a6);
        v8 = *(unsigned int *)(a1 + 24);
        v11 = *(_QWORD *)(a1 + 16);
        v13 = *(_DWORD *)(a1 + 24);
      }
      else
      {
        v19 = (char *)v20 - v11;
        sub_CFC2E0(v18, v12, v8, v9, a5, (__int64)a6);
        v11 = *(_QWORD *)(a1 + 16);
        v8 = *(unsigned int *)(a1 + 24);
        v10 = &v19[v11];
        v13 = *(_DWORD *)(a1 + 24);
      }
    }
    v14 = v11 + 32 * v8;
    if ( v14 )
    {
      *(_QWORD *)v14 = 4;
      v15 = *((_QWORD *)v10 + 2);
      *(_QWORD *)(v14 + 8) = 0;
      *(_QWORD *)(v14 + 16) = v15;
      if ( v15 != 0 && v15 != -4096 && v15 != -8192 )
        sub_BD6050((unsigned __int64 *)v14, *(_QWORD *)v10 & 0xFFFFFFFFFFFFFFF8LL);
      *(_DWORD *)(v14 + 24) = *((_DWORD *)v10 + 6);
      v13 = *(_DWORD *)(a1 + 24);
    }
    v16 = v21;
    v17 = (unsigned int)(v13 + 1);
    *(_DWORD *)(a1 + 24) = v17;
    LOBYTE(v9) = v16 != 0;
    LOBYTE(v17) = v16 != -4096;
    if ( ((unsigned __int8)v17 & (v16 != 0)) != 0 && v16 != -8192 )
      sub_BD60C0(v20);
    sub_CFDBA0(a1, a2, v17, v9, a5, a6);
  }
}
