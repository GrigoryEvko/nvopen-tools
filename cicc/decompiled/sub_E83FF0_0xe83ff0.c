// Function: sub_E83FF0
// Address: 0xe83ff0
//
__int64 __fastcall sub_E83FF0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r15
  __int64 v8; // rsi
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r14
  __int64 result; // rax
  __int64 v15; // r12
  __int64 v16; // rbx
  int v17; // edx
  __int64 v18; // rdx
  __int64 v19; // rdi
  __int64 v20; // rbx
  __int64 v21; // rdi
  int v22; // ebx
  __int64 v23; // [rsp+8h] [rbp-48h]
  unsigned __int64 v24[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = a1 + 16;
  v8 = a1 + 16;
  v13 = sub_C8D7D0(a1, a1 + 16, a2, 0x30u, v24, a6);
  result = *(_QWORD *)a1;
  v15 = *(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v15 )
  {
    v16 = v13;
    do
    {
      while ( 1 )
      {
        if ( v16 )
        {
          v17 = *(_DWORD *)result;
          *(_DWORD *)(v16 + 16) = 0;
          *(_DWORD *)(v16 + 20) = 3;
          *(_DWORD *)v16 = v17;
          *(_QWORD *)(v16 + 8) = v16 + 24;
          v18 = *(unsigned int *)(result + 16);
          if ( (_DWORD)v18 )
            break;
        }
        result += 48;
        v16 += 48;
        if ( v15 == result )
          goto LABEL_7;
      }
      v8 = result + 8;
      v19 = v16 + 8;
      v23 = result;
      v16 += 48;
      sub_E83220(v19, (char **)(result + 8), v18, v10, v11, v12);
      result = v23 + 48;
    }
    while ( v15 != v23 + 48 );
LABEL_7:
    result = *(unsigned int *)(a1 + 8);
    v20 = *(_QWORD *)a1;
    v15 = *(_QWORD *)a1 + 48 * result;
    if ( *(_QWORD *)a1 != v15 )
    {
      do
      {
        v15 -= 48;
        v21 = *(_QWORD *)(v15 + 8);
        result = v15 + 24;
        if ( v21 != v15 + 24 )
          result = _libc_free(v21, v8);
      }
      while ( v15 != v20 );
      v15 = *(_QWORD *)a1;
    }
  }
  v22 = v24[0];
  if ( v7 != v15 )
    result = _libc_free(v15, v8);
  *(_QWORD *)a1 = v13;
  *(_DWORD *)(a1 + 12) = v22;
  return result;
}
