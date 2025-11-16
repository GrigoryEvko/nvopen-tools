// Function: sub_25FCDB0
// Address: 0x25fcdb0
//
__int64 **__fastcall sub_25FCDB0(__int64 **a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v5; // rdi
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // r11
  __int64 *v10; // rcx
  int v12; // edx
  __int64 *v13; // rdx
  __int64 *v14; // rax
  int v15; // ebx

  v3 = *((unsigned int *)a2 + 6);
  v5 = a2[1];
  if ( (_DWORD)v3 )
  {
    v7 = (v3 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v8 = (__int64 *)(v5 + 16LL * v7);
    v9 = *v8;
    if ( *v8 == a3 )
    {
LABEL_3:
      v10 = (__int64 *)*a2;
      *a1 = a2;
      a1[2] = v8;
      a1[3] = (__int64 *)(v5 + 16 * v3);
      a1[1] = v10;
      return a1;
    }
    v12 = 1;
    while ( v9 != -4096 )
    {
      v15 = v12 + 1;
      v7 = (v3 - 1) & (v12 + v7);
      v8 = (__int64 *)(v5 + 16LL * v7);
      v9 = *v8;
      if ( *v8 == a3 )
        goto LABEL_3;
      v12 = v15;
    }
  }
  v13 = (__int64 *)*a2;
  *a1 = a2;
  v14 = (__int64 *)(v5 + 16 * v3);
  a1[2] = v14;
  a1[3] = v14;
  a1[1] = v13;
  return a1;
}
