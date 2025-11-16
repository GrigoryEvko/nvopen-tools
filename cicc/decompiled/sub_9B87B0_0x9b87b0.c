// Function: sub_9B87B0
// Address: 0x9b87b0
//
__int64 __fastcall sub_9B87B0(char *src, unsigned __int64 a2, __int64 a3)
{
  char *v4; // rsi
  char **v6; // rbx
  unsigned __int64 v7; // r12
  char **v8; // r13
  char **v9; // rax
  size_t v10; // r12
  __int64 v11; // rdi
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rbx
  __int64 v14; // rax
  __int64 result; // rax
  _BYTE *v16; // rdi
  _QWORD v18[2]; // [rsp+20h] [rbp-D0h] BYREF
  _BYTE v19[64]; // [rsp+30h] [rbp-C0h] BYREF
  _QWORD v20[2]; // [rsp+70h] [rbp-80h] BYREF
  _BYTE v21[112]; // [rsp+80h] [rbp-70h] BYREF

  v4 = v21;
  v18[0] = v19;
  v18[1] = 0x1000000000LL;
  v20[0] = v21;
  v20[1] = 0x1000000000LL;
  if ( a2 > 1 )
  {
    v6 = (char **)v20;
    LODWORD(v7) = 2;
    v8 = (char **)v18;
    do
    {
      while ( 1 )
      {
        v4 = src;
        if ( !(unsigned __int8)sub_9B8470(v7, src, a2, (__int64)v8) )
          break;
        v9 = v8;
        src = *v8;
        a2 = *((unsigned int *)v8 + 2);
        v8 = v6;
        v6 = v9;
      }
      v7 = (unsigned int)(v7 + 1);
    }
    while ( v7 <= a2 );
  }
  v10 = 4 * a2;
  v11 = 0;
  v12 = *(unsigned int *)(a3 + 12);
  v13 = (__int64)(4 * a2) >> 2;
  *(_DWORD *)(a3 + 8) = 0;
  v14 = 0;
  if ( v13 > v12 )
  {
    v4 = (char *)(a3 + 16);
    sub_C8D5F0(a3, a3 + 16, v13, 4);
    v14 = *(unsigned int *)(a3 + 8);
    v11 = 4 * v14;
  }
  if ( v10 )
  {
    v4 = src;
    memcpy((void *)(*(_QWORD *)a3 + v11), src, v10);
    v14 = *(unsigned int *)(a3 + 8);
  }
  result = v13 + v14;
  v16 = (_BYTE *)v20[0];
  *(_DWORD *)(a3 + 8) = result;
  if ( v16 != v21 )
    result = _libc_free(v16, v4);
  if ( (_BYTE *)v18[0] != v19 )
    return _libc_free(v18[0], v4);
  return result;
}
