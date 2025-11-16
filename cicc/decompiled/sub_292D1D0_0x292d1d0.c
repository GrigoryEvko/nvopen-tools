// Function: sub_292D1D0
// Address: 0x292d1d0
//
void __fastcall sub_292D1D0(__int64 a1, unsigned int a2)
{
  __int64 *v3; // r13
  char v4; // dl
  unsigned __int64 v5; // rax
  unsigned int v6; // ebx
  __int64 v7; // r14
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 *v12; // r14
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // [rsp+0h] [rbp-40h] BYREF
  __int64 v18; // [rsp+8h] [rbp-38h]
  __int64 v19; // [rsp+10h] [rbp-30h]
  __int64 v20; // [rsp+18h] [rbp-28h]
  _BYTE v21[32]; // [rsp+20h] [rbp-20h] BYREF

  v3 = *(__int64 **)(a1 + 16);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 > 1 )
  {
    v5 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
            | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
            | (a2 - 1)
            | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
          | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 16)
        | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
        | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1))
       + 1;
    v6 = v5;
    if ( (unsigned int)v5 > 0x40 )
    {
      if ( !v4 )
      {
        v7 = *(unsigned int *)(a1 + 24);
        v8 = 32LL * (unsigned int)v5;
        goto LABEL_5;
      }
      if ( v3 == (__int64 *)-4096LL )
      {
        v14 = 32LL * (unsigned int)v5;
        v12 = &v17;
        goto LABEL_18;
      }
      if ( v3 == (__int64 *)-8192LL )
      {
        v12 = &v17;
      }
      else
      {
        v11 = *(_QWORD *)(a1 + 24);
        v17 = *(_QWORD *)(a1 + 16);
        v12 = (__int64 *)v21;
        v18 = v11;
        v19 = *(_QWORD *)(a1 + 32);
        v20 = *(_QWORD *)(a1 + 40);
      }
    }
    else
    {
      if ( !v4 )
      {
        v7 = *(unsigned int *)(a1 + 24);
        v6 = 64;
        v8 = 2048;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v6;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_6:
        v10 = 4 * v7;
        sub_292D010(a1, v3, &v3[v10]);
        sub_C7D6A0((__int64)v3, v10 * 8, 8);
        return;
      }
      if ( v3 == (__int64 *)-4096LL || v3 == (__int64 *)-8192LL )
      {
        v14 = 2048;
        v6 = 64;
        v12 = &v17;
        goto LABEL_18;
      }
      v13 = *(_QWORD *)(a1 + 24);
      v17 = *(_QWORD *)(a1 + 16);
      v12 = (__int64 *)v21;
      v6 = 64;
      v18 = v13;
      v19 = *(_QWORD *)(a1 + 32);
      v20 = *(_QWORD *)(a1 + 40);
    }
    v14 = 32LL * v6;
LABEL_18:
    *(_BYTE *)(a1 + 8) &= ~1u;
    v15 = sub_C7D670(v14, 8);
    *(_DWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 16) = v15;
    goto LABEL_19;
  }
  if ( !v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_BYTE *)(a1 + 8) |= 1u;
    goto LABEL_6;
  }
  if ( v3 == (__int64 *)-4096LL || v3 == (__int64 *)-8192LL )
  {
    v12 = &v17;
  }
  else
  {
    v16 = *(_QWORD *)(a1 + 24);
    v17 = *(_QWORD *)(a1 + 16);
    v12 = (__int64 *)v21;
    v18 = v16;
    v19 = *(_QWORD *)(a1 + 32);
    v20 = *(_QWORD *)(a1 + 40);
  }
LABEL_19:
  sub_292D010(a1, &v17, v12);
}
