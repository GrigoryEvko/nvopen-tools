// Function: sub_227BBF0
// Address: 0x227bbf0
//
__int64 **__fastcall sub_227BBF0(__int64 **a1, __int64 *a2, __int64 *a3)
{
  __int64 v4; // rdx
  __int64 v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // r9
  int v9; // r12d
  unsigned __int64 v10; // rax
  unsigned int i; // eax
  __int64 *v12; // r10
  unsigned int v13; // eax
  __int64 v14; // rax
  __int64 *v15; // rdx
  __int64 *v16; // rax
  __int64 *v18; // rax

  v4 = *((unsigned int *)a2 + 6);
  v6 = a2[1];
  if ( (_DWORD)v4 )
  {
    v7 = *a3;
    v8 = a3[1];
    v9 = 1;
    v10 = 0xBF58476D1CE4E5B9LL
        * (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)
         | ((unsigned __int64)(((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4)) << 32));
    for ( i = (v4 - 1) & ((v10 >> 31) ^ v10); ; i = (v4 - 1) & v13 )
    {
      v12 = (__int64 *)(v6 + 24LL * i);
      if ( *v12 == v7 && v12[1] == v8 )
        break;
      if ( *v12 == -4096 && v12[1] == -4096 )
        goto LABEL_7;
      v13 = v9 + i;
      ++v9;
    }
    v18 = (__int64 *)*a2;
    *a1 = a2;
    a1[2] = v12;
    a1[1] = v18;
    a1[3] = (__int64 *)(v6 + 24 * v4);
    return a1;
  }
  else
  {
LABEL_7:
    v14 = 3 * v4;
    v15 = (__int64 *)*a2;
    *a1 = a2;
    v16 = (__int64 *)(v6 + 8 * v14);
    a1[2] = v16;
    a1[3] = v16;
    a1[1] = v15;
    return a1;
  }
}
