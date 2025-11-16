// Function: sub_13214E0
// Address: 0x13214e0
//
__int64 __fastcall sub_13214E0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 *a5,
        const __m128i *a6,
        __int64 a7)
{
  __m128i v8; // xmm1
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned __int64 v13; // rdi
  char *v14; // rbx
  char *v15; // rcx
  unsigned int v16; // ebx
  unsigned int v17; // ebx
  unsigned int v18; // eax
  __int64 v19; // rsi
  __int64 v20; // [rsp+8h] [rbp-38h] BYREF
  _OWORD v21[3]; // [rsp+10h] [rbp-30h] BYREF

  if ( a5 == 0 || a6 == 0 || a7 != 32 || !a4 )
    return 22;
  v8 = _mm_loadu_si128(a6 + 1);
  v21[0] = _mm_loadu_si128(a6);
  v21[1] = v8;
  v10 = sub_1346BB0(a1, v21);
  v20 = v10;
  if ( !v10 )
    return 11;
  v11 = *a5;
  if ( *a5 == 8 )
  {
    *a4 = v10;
    return 0;
  }
  else
  {
    if ( (unsigned __int64)*a5 > 8 )
      v11 = 8;
    if ( (unsigned int)v11 >= 8 )
    {
      *a4 = v10;
      v13 = (unsigned __int64)(a4 + 1) & 0xFFFFFFFFFFFFFFF8LL;
      *(__int64 *)((char *)a4 + (unsigned int)v11 - 8) = *(__int64 *)((char *)&v20 + (unsigned int)v11 - 8);
      v14 = (char *)a4 - v13;
      v15 = (char *)((char *)&v20 - v14);
      v16 = (v11 + (_DWORD)v14) & 0xFFFFFFF8;
      if ( v16 >= 8 )
      {
        v17 = v16 & 0xFFFFFFF8;
        v18 = 0;
        do
        {
          v19 = v18;
          v18 += 8;
          *(_QWORD *)(v13 + v19) = *(_QWORD *)&v15[v19];
        }
        while ( v18 < v17 );
      }
    }
    else if ( (v11 & 4) != 0 )
    {
      *(_DWORD *)a4 = v20;
      *(_DWORD *)((char *)a4 + (unsigned int)v11 - 4) = *(_DWORD *)((char *)&v20 + (unsigned int)v11 - 4);
    }
    else if ( (_DWORD)v11 )
    {
      *(_BYTE *)a4 = v20;
      if ( (v11 & 2) != 0 )
        *(_WORD *)((char *)a4 + (unsigned int)v11 - 2) = *(_WORD *)((char *)&v20 + (unsigned int)v11 - 2);
    }
    *a5 = v11;
    return 22;
  }
}
