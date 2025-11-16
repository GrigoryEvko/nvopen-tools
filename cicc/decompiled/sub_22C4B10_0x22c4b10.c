// Function: sub_22C4B10
// Address: 0x22c4b10
//
__int64 __fastcall sub_22C4B10(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r12
  bool v7; // zf
  unsigned __int64 *v8; // rbx
  unsigned __int64 v9; // r13
  unsigned __int64 *v10; // r13
  __int64 v11; // rax
  __int64 result; // rax
  __int64 v13; // rbx
  int v14; // edx
  unsigned int v15; // eax
  __int64 v16; // r13
  __int64 v17; // rsi
  unsigned __int8 v18; // al
  int v19; // eax
  int v20; // eax
  int v21; // eax
  int v22; // edx
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  __int64 v25; // rax
  int v26; // edx
  int v27; // ecx
  int v28; // r8d
  __int64 v29; // rdi
  __int64 v30; // rax
  _QWORD v31[2]; // [rsp+0h] [rbp-70h] BYREF
  __int64 v32; // [rsp+10h] [rbp-60h]
  __int64 v33; // [rsp+20h] [rbp-50h] BYREF
  __int64 v34; // [rsp+28h] [rbp-48h]
  __int64 v35; // [rsp+30h] [rbp-40h]

  v6 = a2;
  v7 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v33 = 0;
  v34 = 0;
  v35 = -4096;
  if ( v7 )
  {
    v8 = *(unsigned __int64 **)(a1 + 16);
    v9 = (unsigned __int64)*(unsigned int *)(a1 + 24) << 6;
  }
  else
  {
    v8 = (unsigned __int64 *)(a1 + 16);
    v9 = 256;
  }
  v10 = (unsigned __int64 *)((char *)v8 + v9);
  if ( v8 != v10 )
  {
    do
    {
      if ( v8 )
      {
        *v8 = 0;
        v8[1] = 0;
        v11 = v35;
        v7 = v35 == 0;
        v8[2] = v35;
        if ( v11 != -4096 && !v7 && v11 != -8192 )
          sub_BD6050(v8, v33 & 0xFFFFFFFFFFFFFFF8LL);
      }
      v8 += 8;
    }
    while ( v10 != v8 );
    if ( v35 != 0 && v35 != -4096 && v35 != -8192 )
      sub_BD60C0(&v33);
  }
  v31[0] = 0;
  result = -4096;
  v31[1] = 0;
  v32 = -4096;
  v33 = 0;
  v34 = 0;
  v35 = -8192;
  if ( a2 != a3 )
  {
    while ( 1 )
    {
      v13 = *(_QWORD *)(v6 + 16);
      if ( v13 != result && v35 != v13 )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          a4 = a1 + 16;
          v14 = 3;
        }
        else
        {
          v20 = *(_DWORD *)(a1 + 24);
          a4 = *(_QWORD *)(a1 + 16);
          if ( !v20 )
            BUG();
          v14 = v20 - 1;
        }
        v15 = v14 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v16 = a4 + ((unsigned __int64)v15 << 6);
        v17 = *(_QWORD *)(v16 + 16);
        if ( v13 != v17 )
        {
          v28 = 1;
          v29 = 0;
          while ( v17 != -4096 )
          {
            if ( !v29 && v17 == -8192 )
              v29 = v16;
            v15 = v14 & (v28 + v15);
            v16 = a4 + ((unsigned __int64)v15 << 6);
            v17 = *(_QWORD *)(v16 + 16);
            if ( v13 == v17 )
              goto LABEL_28;
            ++v28;
          }
          if ( v29 )
          {
            v30 = *(_QWORD *)(v29 + 16);
            v16 = v29;
          }
          else
          {
            v30 = *(_QWORD *)(v16 + 16);
          }
          if ( v13 != v30 )
          {
            if ( v30 != -4096 && v30 != 0 && v30 != -8192 )
              sub_BD60C0((_QWORD *)v16);
            *(_QWORD *)(v16 + 16) = v13;
            if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
              sub_BD73F0(v16);
          }
        }
LABEL_28:
        v18 = *(_BYTE *)(v6 + 24);
        *(_WORD *)(v16 + 24) = v18;
        if ( v18 <= 3u )
        {
          if ( v18 > 1u )
            *(_QWORD *)(v16 + 32) = *(_QWORD *)(v6 + 32);
        }
        else if ( (unsigned __int8)(v18 - 4) <= 1u )
        {
          *(_DWORD *)(v16 + 40) = *(_DWORD *)(v6 + 40);
          *(_QWORD *)(v16 + 32) = *(_QWORD *)(v6 + 32);
          v19 = *(_DWORD *)(v6 + 56);
          *(_DWORD *)(v6 + 40) = 0;
          *(_DWORD *)(v16 + 56) = v19;
          *(_QWORD *)(v16 + 48) = *(_QWORD *)(v6 + 48);
          LOBYTE(v19) = *(_BYTE *)(v6 + 25);
          *(_DWORD *)(v6 + 56) = 0;
          *(_BYTE *)(v16 + 25) = v19;
        }
        *(_BYTE *)(v6 + 24) = 0;
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        if ( (unsigned int)*(unsigned __int8 *)(v6 + 24) - 4 <= 1 )
        {
          if ( *(_DWORD *)(v6 + 56) > 0x40u )
          {
            v23 = *(_QWORD *)(v6 + 48);
            if ( v23 )
              j_j___libc_free_0_0(v23);
          }
          if ( *(_DWORD *)(v6 + 40) > 0x40u )
          {
            v24 = *(_QWORD *)(v6 + 32);
            if ( v24 )
              j_j___libc_free_0_0(v24);
          }
        }
        v13 = *(_QWORD *)(v6 + 16);
      }
      if ( v13 != -4096 && v13 != 0 && v13 != -8192 )
        sub_BD60C0((_QWORD *)v6);
      v6 += 64;
      if ( a3 == v6 )
        break;
      result = v32;
    }
    v21 = v35;
    if ( v35 != 0 && v35 != -4096 && v35 != -8192 )
    {
      v25 = sub_BD60C0(&v33);
      v26 = v32;
      LOBYTE(v25) = v32 != -8192;
      LOBYTE(v27) = v32 != 0;
      LOBYTE(v26) = v32 != -4096;
      result = v26 & v27 & (unsigned int)v25;
    }
    else
    {
      v22 = v32;
      LOBYTE(v21) = v32 != -8192;
      LOBYTE(a4) = v32 != -4096;
      LOBYTE(v22) = v32 != 0;
      result = v22 & (unsigned int)a4 & v21;
    }
    if ( (_BYTE)result )
      return sub_BD60C0(v31);
  }
  return result;
}
