// Function: sub_2B24BA0
// Address: 0x2b24ba0
//
__int64 *__fastcall sub_2B24BA0(__int64 *a1, __int64 a2, unsigned int a3, int a4)
{
  int v8; // ebx
  __int64 v9; // rax
  unsigned int v10; // edx
  unsigned int v11; // esi
  __int64 v12; // rax
  __int64 v13; // rdi
  int v14; // ecx
  unsigned __int64 v15; // r8
  char v16; // cl
  __int64 v17; // r14
  char v18; // cl
  char v20; // r14
  __int64 *v21; // r10
  unsigned int v22; // r9d
  unsigned int i; // ecx
  __int64 v24; // [rsp+8h] [rbp-38h]

  v8 = 1;
  v9 = *(_QWORD *)(*(_QWORD *)a2 + 8LL);
  if ( *(_BYTE *)(v9 + 8) == 17 )
    v8 = *(_DWORD *)(v9 + 32);
  sub_B48880(a1, a3 * v8, 0);
  v10 = v8;
  v11 = 0;
  v12 = 0;
  v13 = a3;
  if ( a3 )
  {
    do
    {
      while ( 1 )
      {
        v14 = **(unsigned __int8 **)(a2 + 8 * v12);
        if ( (_BYTE)v14 != 13 && v14 - 29 == a4 && v11 != v10 )
          break;
LABEL_5:
        ++v12;
        v10 += v8;
        v11 += v8;
        if ( v13 == v12 )
          return a1;
      }
      v15 = *a1;
      if ( (*a1 & 1) == 0 )
      {
        v20 = v11 & 0x3F;
        v21 = (__int64 *)(*(_QWORD *)v15 + 8LL * (v11 >> 6));
        v24 = *v21;
        if ( v11 >> 6 == v10 >> 6 )
        {
          *v21 = ((1LL << v10) - (1LL << v20)) | v24;
        }
        else
        {
          *v21 = (-1LL << v20) | v24;
          v22 = ((unsigned int)((v11 - (unsigned __int64)(v11 != 0)) >> 6) + (v11 != 0)) << 6;
          for ( i = v22 + 64; v10 >= i; i += 64 )
          {
            *(_QWORD *)(*(_QWORD *)v15 + 8LL * ((i - 64) >> 6)) = -1;
            v22 = i;
          }
          if ( v10 > v22 )
            *(_QWORD *)(*(_QWORD *)v15 + 8LL * (v22 >> 6)) |= (1LL << v10) - 1;
        }
        goto LABEL_5;
      }
      ++v12;
      v16 = v10;
      v10 += v8;
      v17 = 1LL << v16;
      v18 = v11;
      v11 += v8;
      *a1 = 2
          * ((v15 >> 58 << 57) | ~(-1LL << (v15 >> 58)) & (~(-1LL << (v15 >> 58)) & (v15 >> 1) | (v17 - (1LL << v18))))
          + 1;
    }
    while ( v13 != v12 );
  }
  return a1;
}
