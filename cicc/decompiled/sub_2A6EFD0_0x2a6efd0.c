// Function: sub_2A6EFD0
// Address: 0x2a6efd0
//
void __fastcall sub_2A6EFD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdx
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // rbx
  unsigned int v10; // r15d
  unsigned __int8 *v11; // rax
  _BYTE *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdi
  unsigned int v15; // esi
  __int64 *v16; // rcx
  __int64 v17; // r10
  __int64 v18; // rbx
  unsigned __int8 *v19; // rax
  int v20; // ecx
  int v21; // ebx
  int v22; // [rsp-7Ch] [rbp-7Ch]
  __m128i v23; // [rsp-78h] [rbp-78h] BYREF
  unsigned __int8 v24[104]; // [rsp-68h] [rbp-68h] BYREF

  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) == 0 )
    return;
  v5 = *(_QWORD *)(a2 + 40);
  v6 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v7 = *(unsigned int *)(a1 + 272);
  v8 = *(_QWORD *)(v5 + 72);
  if ( !(_DWORD)v7 )
  {
    v5 = *(unsigned int *)(a1 + 320);
    if ( !(_DWORD)v5 )
      return;
    v9 = *(_QWORD *)(v6 + 8);
    if ( *(_BYTE *)(v9 + 8) != 15 )
      return;
    goto LABEL_5;
  }
  v9 = *(_QWORD *)(v6 + 8);
  if ( *(_BYTE *)(v9 + 8) == 15 )
  {
    if ( !*(_DWORD *)(a1 + 320) )
      return;
LABEL_5:
    if ( (unsigned __int8)sub_B19060(a1 + 360, v8, v5, a4) )
    {
      v22 = *(_DWORD *)(v9 + 12);
      if ( v22 )
      {
        v10 = 0;
        do
        {
          v11 = sub_2A6A1C0(a1, (unsigned __int8 *)v6, v10);
          sub_22C05A0((__int64)v24, v11);
          v23.m128i_i32[2] = v10;
          v23.m128i_i64[0] = v8;
          ++v10;
          v12 = (_BYTE *)sub_2A6EC30(a1 + 280, &v23);
          sub_2A639B0(a1, v12, v8, (__int64)v24, 0x100000000LL);
          sub_22C0090(v24);
        }
        while ( v10 != v22 );
      }
    }
    return;
  }
  v13 = *(unsigned int *)(a1 + 256);
  v14 = *(_QWORD *)(a1 + 240);
  if ( (_DWORD)v13 )
  {
    v15 = (v13 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v16 = (__int64 *)(v14 + 16LL * v15);
    v17 = *v16;
    if ( v8 == *v16 )
    {
LABEL_16:
      if ( v16 != (__int64 *)(v14 + 16 * v13) )
      {
        v18 = *(_QWORD *)(a1 + 264) + 48LL * *((unsigned int *)v16 + 2);
        if ( v18 != *(_QWORD *)(a1 + 264) + 48 * v7 )
        {
          v19 = (unsigned __int8 *)sub_2A68BC0(a1, (unsigned __int8 *)v6);
          sub_22C05A0((__int64)v24, v19);
          sub_2A639B0(a1, (_BYTE *)(v18 + 8), v8, (__int64)v24, 0x100000000LL);
          sub_22C0090(v24);
        }
      }
    }
    else
    {
      v20 = 1;
      while ( v17 != -4096 )
      {
        v21 = v20 + 1;
        v15 = (v13 - 1) & (v20 + v15);
        v16 = (__int64 *)(v14 + 16LL * v15);
        v17 = *v16;
        if ( v8 == *v16 )
          goto LABEL_16;
        v20 = v21;
      }
    }
  }
}
