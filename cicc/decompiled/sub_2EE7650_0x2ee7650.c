// Function: sub_2EE7650
// Address: 0x2ee7650
//
__int64 __fastcall sub_2EE7650(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v4; // r9
  char v7; // r8
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // r10
  __int64 *v11; // rdx
  __int64 v12; // r11
  __int64 *v13; // rax
  __int64 v14; // r8
  _QWORD *v15; // rax
  __int64 v16; // r8
  _QWORD *v17; // rax
  unsigned int v19; // edx
  int v20; // eax
  int v21; // r11d
  unsigned int v22; // ebx

  v4 = a3;
  v7 = *(_BYTE *)(a1 + 120);
  v8 = 11LL * *(int *)(a4 + 24);
  v9 = *(_QWORD *)a1 + 8 * v8;
  if ( !v7 )
  {
    if ( *(_DWORD *)(v9 + 24) != -1 )
      return *(unsigned __int8 *)(a1 + 120);
LABEL_3:
    if ( (_BYTE)v4 )
    {
      v9 = *(_QWORD *)(a1 + 112);
      v4 = *(unsigned int *)(v9 + 24);
      v10 = *(_QWORD *)(v9 + 8);
      if ( (_DWORD)v4 )
      {
        v4 = (unsigned int)(v4 - 1);
        v8 = (unsigned int)v4 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v11 = (__int64 *)(v10 + 16 * v8);
        v12 = *v11;
        if ( a2 == *v11 )
        {
LABEL_6:
          v9 = v11[1];
          if ( v9 )
          {
            if ( v7 )
              a2 = a4;
            if ( **(_QWORD **)(v9 + 32) == a2 )
              goto LABEL_19;
            v8 = (unsigned int)v4 & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
            v13 = (__int64 *)(v10 + 16 * v8);
            v14 = *v13;
            if ( a4 != *v13 )
            {
              v20 = 1;
              while ( v14 != -4096 )
              {
                v21 = v20 + 1;
                v8 = (unsigned int)v4 & (v20 + (_DWORD)v8);
                v13 = (__int64 *)(v10 + 16LL * (unsigned int)v8);
                v14 = *v13;
                if ( a4 == *v13 )
                  goto LABEL_11;
                v20 = v21;
              }
              goto LABEL_19;
            }
LABEL_11:
            v15 = (_QWORD *)v13[1];
            if ( (_QWORD *)v9 != v15 )
            {
              while ( v15 )
              {
                v15 = (_QWORD *)*v15;
                if ( (_QWORD *)v9 == v15 )
                  goto LABEL_14;
              }
              goto LABEL_19;
            }
          }
        }
        else
        {
          v9 = 1;
          while ( v12 != -4096 )
          {
            v22 = v9 + 1;
            v8 = (unsigned int)v4 & ((_DWORD)v9 + (_DWORD)v8);
            v11 = (__int64 *)(v10 + 16LL * (unsigned int)v8);
            v12 = *v11;
            if ( a2 == *v11 )
              goto LABEL_6;
            v9 = v22;
          }
        }
      }
    }
LABEL_14:
    v16 = *(unsigned __int8 *)(a1 + 44);
    if ( !(_BYTE)v16 )
    {
LABEL_23:
      sub_C8CC70(a1 + 16, a4, v9, v8, v16, v4);
      return v19;
    }
    v17 = *(_QWORD **)(a1 + 24);
    v8 = *(unsigned int *)(a1 + 36);
    v9 = (__int64)&v17[v8];
    if ( v17 == (_QWORD *)v9 )
    {
LABEL_25:
      if ( (unsigned int)v8 < *(_DWORD *)(a1 + 32) )
      {
        *(_DWORD *)(a1 + 36) = v8 + 1;
        *(_QWORD *)v9 = a4;
        ++*(_QWORD *)(a1 + 16);
        return (unsigned int)v16;
      }
      goto LABEL_23;
    }
    while ( a4 != *v17 )
    {
      if ( (_QWORD *)v9 == ++v17 )
        goto LABEL_25;
    }
LABEL_19:
    LODWORD(v16) = 0;
    return (unsigned int)v16;
  }
  if ( *(_DWORD *)(v9 + 28) == -1 )
    goto LABEL_3;
  return 0;
}
