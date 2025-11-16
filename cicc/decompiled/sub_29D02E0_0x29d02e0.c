// Function: sub_29D02E0
// Address: 0x29d02e0
//
__int64 __fastcall sub_29D02E0(__int64 *a1)
{
  unsigned __int64 v1; // r12
  __int64 v2; // rbx
  char v3; // al
  unsigned int v4; // r13d
  _QWORD *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // rax
  unsigned int v12; // r14d
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r9
  __int64 v17; // rsi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // r8
  int v20; // eax
  __int64 v21; // r8
  __int64 *v22; // rdi
  __int64 *v23; // rsi
  __int64 v24; // rax
  unsigned int v25; // r8d
  unsigned __int64 v27; // rax
  __int64 v28; // [rsp+8h] [rbp-58h]
  __int64 v30; // [rsp+18h] [rbp-48h]
  __int64 v31[7]; // [rsp+28h] [rbp-38h] BYREF

  v1 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v2 = *(_QWORD *)(v1 + 8);
  v3 = *(_BYTE *)(v2 + 8);
  if ( v3 == 17 || v3 == 16 )
  {
    v4 = *(_DWORD *)(v2 + 32);
  }
  else
  {
    v25 = 0;
    if ( v3 != 15 )
      return v25;
    v4 = *(_DWORD *)(v2 + 12);
  }
  v5 = (_QWORD *)sub_22077B0(0x48u);
  v10 = (unsigned __int64)v5;
  if ( v5 )
  {
    *v5 = v2;
    v5[1] = v5 + 3;
    v5[2] = 0x600000000LL;
    v11 = 6;
  }
  else
  {
    v11 = MEMORY[0x14];
  }
  if ( v4 > v11 )
    sub_29D0220(v10 + 8, v4, v6, v7, v8, v9);
  v12 = 0;
  v30 = v10 + 8;
  if ( v4 )
  {
    do
    {
      v13 = sub_AD69F0((unsigned __int8 *)v1, v12);
      v17 = *(unsigned int *)(v10 + 16);
      v18 = *(unsigned int *)(v10 + 20);
      v19 = v17 + 1;
      v31[0] = v13 & 0xFFFFFFFFFFFFFFFBLL;
      v20 = v17;
      if ( v17 + 1 > v18 )
      {
        v27 = *(_QWORD *)(v10 + 8);
        if ( v27 > (unsigned __int64)v31 || (unsigned __int64)v31 >= v27 + 8 * v17 )
        {
          sub_29D0220(v30, v19, v14, v15, v19, v16);
          v17 = *(unsigned int *)(v10 + 16);
          v21 = *(_QWORD *)(v10 + 8);
          v22 = v31;
          v20 = *(_DWORD *)(v10 + 16);
        }
        else
        {
          v28 = *(_QWORD *)(v10 + 8);
          sub_29D0220(v30, v19, v14, v15, v19, v16);
          v21 = *(_QWORD *)(v10 + 8);
          v17 = *(unsigned int *)(v10 + 16);
          v22 = (__int64 *)((char *)v31 + v21 - v28);
          v20 = *(_DWORD *)(v10 + 16);
        }
      }
      else
      {
        v21 = *(_QWORD *)(v10 + 8);
        v22 = v31;
      }
      v23 = (__int64 *)(v21 + 8 * v17);
      if ( v23 )
      {
        *v23 = 0;
        v24 = *v22;
        *v22 = 0;
        *v23 = v24;
        v20 = *(_DWORD *)(v10 + 16);
      }
      ++v12;
      *(_DWORD *)(v10 + 16) = v20 + 1;
      sub_29CF750(v31);
    }
    while ( v4 != v12 );
  }
  v25 = 1;
  *a1 = v10 | 4;
  return v25;
}
