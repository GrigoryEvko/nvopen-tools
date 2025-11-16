// Function: sub_2396EC0
// Address: 0x2396ec0
//
void **__fastcall sub_2396EC0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // r12
  void **result; // rax
  __int64 v8; // rdx
  char *v9; // rsi
  int v10; // r11d
  char *v11; // r8
  int v12; // eax
  char *v13; // r13
  __int64 *v14; // rax
  _QWORD *v15; // rax
  char *v16; // rsi
  _QWORD *v17; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v18; // [rsp+8h] [rbp-A8h]
  __int64 v19; // [rsp+10h] [rbp-A0h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-98h]
  char v21; // [rsp+90h] [rbp-20h] BYREF

  v4 = sub_BC1CD0(a2, &unk_4F82410, a1);
  v5 = *(_QWORD *)(a1 + 40);
  v6 = v4;
  result = *(void ***)(v4 + 8);
  v8 = *((unsigned int *)result + 22);
  v9 = (char *)result[9];
  if ( (_DWORD)v8 )
  {
    v10 = 1;
    for ( result = (void **)(((_DWORD)v8 - 1)
                           & ((unsigned int)((0xBF58476D1CE4E5B9LL
                                            * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)
                                             | ((unsigned __int64)(((unsigned int)&unk_4F86B78 >> 9)
                                                                 ^ ((unsigned int)&unk_4F86B78 >> 4)) << 32))) >> 31)
                            ^ (484763065 * (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)))));
          ;
          result = (void **)(((_DWORD)v8 - 1) & (unsigned int)v12) )
    {
      v11 = &v9[24 * (unsigned int)result];
      if ( *(_UNKNOWN **)v11 == &unk_4F86B78 && v5 == *((_QWORD *)v11 + 1) )
        break;
      if ( *(_QWORD *)v11 == -4096 && *((_QWORD *)v11 + 1) == -4096 )
        return result;
      v12 = v10 + (_DWORD)result;
      ++v10;
    }
    result = (void **)&v9[24 * v8];
    if ( v11 != (char *)result )
    {
      result = (void **)*((_QWORD *)v11 + 2);
      v13 = (char *)result[3];
      if ( v13 )
      {
        v18 = 1;
        v14 = &v19;
        do
        {
          *v14 = -4096;
          v14 += 2;
        }
        while ( v14 != (__int64 *)&v21 );
        if ( (v18 & 1) == 0 )
          sub_C7D6A0(v19, 16LL * v20, 8);
        v15 = (_QWORD *)sub_22077B0(0x10u);
        if ( v15 )
        {
          v15[1] = v13 + 8;
          *v15 = &unk_49DDA38;
        }
        v17 = v15;
        v16 = (char *)a3[2];
        if ( v16 == (char *)a3[3] )
        {
          sub_CF7D50(a3 + 1, v16, &v17);
        }
        else
        {
          if ( v16 )
          {
            *(_QWORD *)v16 = v15;
            v16 = (char *)a3[2];
          }
          a3[2] = (__int64)(v16 + 8);
        }
        return sub_2396AE0(v6 + 8);
      }
    }
  }
  return result;
}
