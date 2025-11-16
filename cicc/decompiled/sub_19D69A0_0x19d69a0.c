// Function: sub_19D69A0
// Address: 0x19d69a0
//
__int64 __fastcall sub_19D69A0(__int64 *a1, __int64 *a2)
{
  int v4; // eax
  __int64 v5; // r11
  __int64 v6; // r10
  __int64 v7; // r9
  char v8; // r8
  __int64 v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // r15
  __int64 v13; // r14
  int v14; // edi
  __int64 v15; // r13
  __int64 v16; // rdi
  bool v17; // cc
  __int64 v18; // rdi
  int v19; // edi
  __int64 v20; // rdi
  __int64 v21; // rdi
  __int64 v23; // [rsp+0h] [rbp-70h]
  __int64 v24; // [rsp+8h] [rbp-68h]
  int v25; // [rsp+18h] [rbp-58h]
  char v26; // [rsp+1Fh] [rbp-51h]
  __int64 v27; // [rsp+20h] [rbp-50h]
  __int64 v28; // [rsp+28h] [rbp-48h]
  __int64 v29; // [rsp+28h] [rbp-48h]
  __int64 v30; // [rsp+30h] [rbp-40h]
  int v31; // [rsp+30h] [rbp-40h]
  unsigned int v32; // [rsp+38h] [rbp-38h]
  int v33; // [rsp+3Ch] [rbp-34h]

  v4 = *((_DWORD *)a1 + 14);
  v5 = *a1;
  *((_DWORD *)a1 + 14) = 0;
  v6 = a1[1];
  v7 = a1[2];
  v8 = *((_BYTE *)a1 + 24);
  v9 = a1[4];
  v10 = a1[5];
  v11 = a1[6];
  v12 = a1[8];
  v13 = a1[9];
  v14 = *((_DWORD *)a1 + 22);
  v15 = a1[10];
  *((_DWORD *)a1 + 22) = 0;
  v33 = v14;
  v32 = *((_DWORD *)a1 + 24);
  *a1 = *a2;
  a1[1] = a2[1];
  a1[2] = a2[2];
  *((_BYTE *)a1 + 24) = *((_BYTE *)a2 + 24);
  a1[4] = a2[4];
  a1[5] = a2[5];
  a1[6] = a2[6];
  *((_DWORD *)a1 + 14) = *((_DWORD *)a2 + 14);
  v16 = a2[8];
  *((_DWORD *)a2 + 14) = 0;
  v17 = *((_DWORD *)a1 + 22) <= 0x40u;
  a1[8] = v16;
  a1[9] = a2[9];
  if ( !v17 )
  {
    v18 = a1[10];
    if ( v18 )
    {
      v23 = v11;
      v25 = v4;
      v24 = v10;
      v26 = v8;
      v27 = v7;
      v28 = v6;
      v30 = v5;
      j_j___libc_free_0_0(v18);
      v11 = v23;
      v4 = v25;
      v10 = v24;
      v8 = v26;
      v7 = v27;
      v6 = v28;
      v5 = v30;
    }
  }
  a1[10] = a2[10];
  *((_DWORD *)a1 + 22) = *((_DWORD *)a2 + 22);
  v19 = *((_DWORD *)a2 + 24);
  *((_DWORD *)a2 + 22) = 0;
  *((_DWORD *)a1 + 24) = v19;
  v17 = *((_DWORD *)a2 + 14) <= 0x40u;
  *a2 = v5;
  a2[1] = v6;
  a2[2] = v7;
  *((_BYTE *)a2 + 24) = v8;
  a2[4] = v9;
  a2[5] = v10;
  if ( !v17 )
  {
    v20 = a2[6];
    if ( v20 )
    {
      v29 = v11;
      v31 = v4;
      j_j___libc_free_0_0(v20);
      v11 = v29;
      v4 = v31;
    }
  }
  v17 = *((_DWORD *)a2 + 22) <= 0x40u;
  a2[6] = v11;
  *((_DWORD *)a2 + 14) = v4;
  a2[8] = v12;
  a2[9] = v13;
  if ( !v17 )
  {
    v21 = a2[10];
    if ( v21 )
      j_j___libc_free_0_0(v21);
  }
  a2[10] = v15;
  *((_DWORD *)a2 + 22) = v33;
  *((_DWORD *)a2 + 24) = v32;
  return v32;
}
