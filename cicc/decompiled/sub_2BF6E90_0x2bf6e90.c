// Function: sub_2BF6E90
// Address: 0x2bf6e90
//
__int64 __fastcall sub_2BF6E90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rbx
  __int64 v8; // r9
  __int64 result; // rax
  unsigned int v10; // r14d
  __int64 v11; // rdx
  __int64 i; // rcx
  unsigned __int64 v13; // rdi
  int v14; // r15d
  _QWORD *v15; // rax
  _QWORD *v16; // [rsp+8h] [rbp-48h]
  unsigned __int64 v17[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = (_QWORD *)(a1 + 112);
  sub_C8CD80(a1, a1 + 32, a2, a4, a5, a6);
  *(_QWORD *)(a1 + 96) = a1 + 112;
  result = 0x800000000LL;
  *(_QWORD *)(a1 + 104) = 0x800000000LL;
  v10 = *(_DWORD *)(a2 + 104);
  if ( v10 )
  {
    result = a2 + 96;
    if ( a1 + 96 != a2 + 96 )
    {
      v11 = v10;
      if ( v10 > 8 )
      {
        v16 = (_QWORD *)sub_C8D7D0(a1 + 96, (__int64)v7, v10, 0x28u, v17, v8);
        sub_2BF6E30(a1 + 96, v16);
        v13 = *(_QWORD *)(a1 + 96);
        v14 = v17[0];
        v15 = v16;
        if ( v7 != (_QWORD *)v13 )
        {
          _libc_free(v13);
          v15 = v16;
        }
        *(_QWORD *)(a1 + 96) = v15;
        v11 = *(unsigned int *)(a2 + 104);
        v7 = v15;
        *(_DWORD *)(a1 + 108) = v14;
      }
      result = *(_QWORD *)(a2 + 96);
      for ( i = result + 40 * v11; i != result; v7 += 5 )
      {
        if ( v7 )
        {
          *v7 = *(_QWORD *)result;
          v7[1] = *(_QWORD *)(result + 8);
          v7[2] = *(_QWORD *)(result + 16);
          v7[3] = *(_QWORD *)(result + 24);
          v7[4] = *(_QWORD *)(result + 32);
        }
        result += 40;
      }
      *(_DWORD *)(a1 + 104) = v10;
    }
  }
  return result;
}
