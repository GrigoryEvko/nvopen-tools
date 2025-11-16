// Function: sub_2981630
// Address: 0x2981630
//
void __fastcall sub_2981630(__int64 *a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // r8
  __int64 v7; // r9
  __int64 v8; // r11
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 *v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // rax
  _BYTE *v19; // rsi
  int v20; // r10d
  __int64 v21[3]; // [rsp+8h] [rbp-18h] BYREF

  v3 = *a1;
  v21[0] = a2;
  v4 = *(unsigned int *)(v3 + 280);
  v5 = *(_QWORD *)(v3 + 264);
  if ( !(_DWORD)v4 )
    return;
  v7 = (unsigned int)(v4 - 1);
  v8 = (unsigned int)v7 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = v5 + 48 * v8;
  v10 = *(_QWORD *)v9;
  if ( a2 == *(_QWORD *)v9 )
  {
LABEL_3:
    if ( v9 == v5 + 48 * v4 )
      return;
    v11 = *(__int64 **)(v9 + 8);
    v12 = 8LL * *(unsigned int *)(v9 + 16);
    v13 = &v11[(unsigned __int64)v12 / 8];
    v14 = v12 >> 3;
    v15 = v12 >> 5;
    if ( v15 )
    {
      v16 = (__int64)&v11[4 * v15];
      while ( 1 )
      {
        v14 = *v11;
        if ( *(_QWORD *)(*v11 + 40) )
          goto LABEL_11;
        v14 = v11[1];
        if ( *(_QWORD *)(v14 + 40) )
        {
          if ( v13 == v11 + 1 )
            return;
          goto LABEL_13;
        }
        v14 = v11[2];
        if ( *(_QWORD *)(v14 + 40) )
        {
          if ( v13 == v11 + 2 )
            return;
          goto LABEL_13;
        }
        v14 = v11[3];
        if ( *(_QWORD *)(v14 + 40) )
        {
          if ( v13 == v11 + 3 )
            return;
          goto LABEL_13;
        }
        v11 += 4;
        if ( v11 == (__int64 *)v16 )
        {
          v14 = v13 - v11;
          break;
        }
      }
    }
    if ( v14 != 2 )
    {
      if ( v14 != 3 )
      {
        if ( v14 != 1 )
          return;
LABEL_21:
        v16 = *v11;
        if ( !*(_QWORD *)(*v11 + 40) )
          return;
        goto LABEL_11;
      }
      v16 = *v11;
      if ( *(_QWORD *)(*v11 + 40) )
      {
LABEL_11:
        if ( v13 != v11 )
        {
LABEL_13:
          v17 = sub_29812B0(v3 + 208, v21, v16, v14, v5, v7);
          v18 = a1[1];
          v19 = *(_BYTE **)(v17 + 8);
          if ( v19 == *(_BYTE **)(v17 + 16) )
          {
            sub_297E8B0(v17, v19, (_QWORD *)(v18 + 32));
          }
          else
          {
            if ( v19 )
            {
              *(_QWORD *)v19 = *(_QWORD *)(v18 + 32);
              v19 = *(_BYTE **)(v17 + 8);
            }
            *(_QWORD *)(v17 + 8) = v19 + 8;
          }
        }
        return;
      }
      ++v11;
    }
    v16 = *v11;
    if ( !*(_QWORD *)(*v11 + 40) )
    {
      ++v11;
      goto LABEL_21;
    }
    goto LABEL_11;
  }
  v20 = 1;
  while ( v10 != -4096 )
  {
    v8 = (unsigned int)v7 & ((_DWORD)v8 + v20);
    v9 = v5 + 48 * v8;
    v10 = *(_QWORD *)v9;
    if ( a2 == *(_QWORD *)v9 )
      goto LABEL_3;
    ++v20;
  }
}
