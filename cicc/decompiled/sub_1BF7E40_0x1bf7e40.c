// Function: sub_1BF7E40
// Address: 0x1bf7e40
//
bool __fastcall sub_1BF7E40(unsigned int a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  unsigned int v8; // r8d
  unsigned int v10; // eax
  unsigned int v11; // ecx
  unsigned int v12; // r15d
  unsigned int v13; // r13d
  unsigned int v14; // r14d
  __int64 v15; // rsi
  unsigned int v16; // ebx
  unsigned __int64 v17; // rax
  unsigned int v18; // ecx
  __int64 v19; // rax
  unsigned int *v20; // rdx
  unsigned __int64 v22; // [rsp+0h] [rbp-40h]

  v8 = a1;
  v10 = a1 % 0xA;
  v11 = *a2;
  v12 = *a2 >> 5;
  if ( v8 > 0x58 || (v13 = 16, !v10) )
    v13 = v10 == 0 ? 32 : 24;
  v14 = 64;
  if ( v8 > 0x45 )
  {
    v14 = 32;
    if ( v8 != 75 )
      v14 = v10 == 0 ? 64 : 48;
  }
  v15 = *(unsigned int *)(a3 + 8);
  v16 = 8;
  while ( 1 )
  {
    v18 = 0x10000 / v11;
    v19 = v18 / v16;
    if ( (unsigned int)v19 > v13 )
      goto LABEL_10;
    if ( (unsigned int)v19 < a2[1] || v18 < v16 )
      return (_DWORD)v15 != 0;
    if ( v12 * (unsigned int)v19 > v14 )
      goto LABEL_10;
    if ( !(_DWORD)v15 || (v20 = (unsigned int *)(*(_QWORD *)a3 + 8LL * (unsigned int)v15 - 8), (_DWORD)v19 != v20[1]) )
    {
      v17 = v16 | (unsigned __int64)(v19 << 32);
      if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v15 )
      {
        v22 = v17;
        sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, v8, a6);
        v17 = v22;
        v15 = *(unsigned int *)(a3 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a3 + 8 * v15) = v17;
      v15 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
      *(_DWORD *)(a3 + 8) = v15;
LABEL_10:
      v16 += 8;
      if ( v16 == 264 )
        return (_DWORD)v15 != 0;
      goto LABEL_11;
    }
    *v20 = v16;
    v16 += 8;
    v15 = *(unsigned int *)(a3 + 8);
    if ( v16 == 264 )
      return (_DWORD)v15 != 0;
LABEL_11:
    v11 = *a2;
  }
}
