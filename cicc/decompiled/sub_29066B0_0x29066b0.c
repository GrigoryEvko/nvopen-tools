// Function: sub_29066B0
// Address: 0x29066b0
//
char *__fastcall sub_29066B0(char **a1, unsigned __int8 **a2)
{
  char *result; // rax
  char v3; // r12
  unsigned __int8 *v4; // r14
  char *v5; // r13
  __int64 v7; // rsi
  unsigned __int8 *v8; // r15
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rdx
  unsigned __int8 **v12; // rcx
  int v13; // edi
  unsigned int v14; // edx
  unsigned __int8 **v15; // rax
  unsigned __int8 *v16; // r8
  int v17; // eax
  int v18; // r9d

  result = *a1;
  v3 = **a1;
  if ( v3 )
  {
    v4 = *a2;
    v5 = a1[1];
    if ( **(unsigned __int8 ***)v5 != sub_BD3990(*a2, (__int64)a2) )
    {
      v7 = *((_QWORD *)v5 + 1);
      v8 = (unsigned __int8 *)sub_2906530((__int64)v4, v7, *((_QWORD *)v5 + 2));
      if ( v8 != sub_BD3990(v4, v7) )
      {
        result = *a1;
        v3 = 0;
        goto LABEL_2;
      }
      v9 = *((_QWORD *)v5 + 3);
      v10 = *(_QWORD *)(v9 + 8);
      v11 = *(unsigned int *)(v9 + 24);
      v12 = (unsigned __int8 **)(v10 + 16 * v11);
      if ( (_DWORD)v11 )
      {
        v13 = v11 - 1;
        v14 = (v11 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v15 = (unsigned __int8 **)(v10 + 16LL * v14);
        v16 = *v15;
        if ( v8 == *v15 )
        {
LABEL_8:
          v3 = v12 == v15;
        }
        else
        {
          v17 = 1;
          while ( v16 != (unsigned __int8 *)-4096LL )
          {
            v18 = v17 + 1;
            v14 = v13 & (v17 + v14);
            v15 = (unsigned __int8 **)(v10 + 16LL * v14);
            v16 = *v15;
            if ( v8 == *v15 )
              goto LABEL_8;
            v17 = v18;
          }
        }
      }
    }
    result = *a1;
  }
LABEL_2:
  *result = v3;
  return result;
}
