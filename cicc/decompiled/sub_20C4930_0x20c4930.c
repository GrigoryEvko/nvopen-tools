// Function: sub_20C4930
// Address: 0x20c4930
//
__int64 __fastcall sub_20C4930(__int64 *a1, __int64 a2, unsigned int a3, unsigned int a4)
{
  __int64 v7; // r8
  __int64 v8; // rcx
  __int64 result; // rax
  _QWORD *v10; // rdi
  unsigned int v11; // ebx
  unsigned int *v12; // rdx
  unsigned int v13; // esi
  __int64 i; // rbx
  __int64 v15; // rdi
  __int64 v16; // [rsp+8h] [rbp-68h]
  char v17[8]; // [rsp+10h] [rbp-60h] BYREF
  int v18; // [rsp+18h] [rbp-58h] BYREF
  __int64 v19; // [rsp+20h] [rbp-50h]
  int *v20; // [rsp+28h] [rbp-48h]
  int *v21; // [rsp+30h] [rbp-40h]
  __int64 v22; // [rsp+38h] [rbp-38h]

  v20 = &v18;
  v21 = &v18;
  v18 = 0;
  v19 = 0;
  v22 = 0;
  sub_20C47C0((__int64)a1, a2, (__int64)v17);
  sub_20C3910(a1, a2, a3, (__int64)v17);
  sub_20C3FC0((__int64)a1, a2, a3);
  v7 = a1[4];
  v8 = a1[9];
  result = *(unsigned int *)(v7 + 16);
  if ( (_DWORD)result )
  {
    v10 = (_QWORD *)a1[9];
    v11 = 0;
    while ( 1 )
    {
      if ( *(_DWORD *)(v10[13] + 4LL * v11) == -1 || *(_DWORD *)(v10[16] + 4LL * v11) != -1 )
      {
        v12 = (unsigned int *)(*(_QWORD *)(v8 + 128) + 4LL * v11);
        result = *v12;
        if ( (unsigned int)result < a4 && (unsigned int)result >= a3 )
        {
          *v12 = a3;
          v7 = a1[4];
        }
        if ( *(_DWORD *)(v7 + 16) == ++v11 )
          break;
      }
      else
      {
        v13 = v11;
        v16 = v8;
        ++v11;
        result = sub_20C2470(v10, v13, 0);
        v7 = a1[4];
        v8 = v16;
        if ( *(_DWORD *)(v7 + 16) == v11 )
          break;
      }
      v10 = (_QWORD *)a1[9];
    }
  }
  for ( i = v19; i; result = j_j___libc_free_0(v15, 40) )
  {
    sub_20C31F0((__int64)v17, *(_QWORD *)(i + 24));
    v15 = i;
    i = *(_QWORD *)(i + 16);
  }
  return result;
}
