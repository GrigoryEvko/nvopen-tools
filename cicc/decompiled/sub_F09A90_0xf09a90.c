// Function: sub_F09A90
// Address: 0xf09a90
//
void __fastcall sub_F09A90(_BYTE *a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rax
  __int64 v8; // rcx
  char v9; // dl
  unsigned __int8 v10; // al
  __int64 v11; // rdx
  _BYTE **v12; // rbx
  _BYTE **v13; // r12
  _BYTE *v14; // rsi
  _QWORD *v15; // rax

  if ( !a1 || (unsigned __int8)(*a1 - 5) > 0x1Fu )
    return;
  if ( !*(_BYTE *)(a2 + 28) )
    goto LABEL_9;
  v7 = *(_QWORD **)(a2 + 8);
  a4 = *(unsigned int *)(a2 + 20);
  a3 = &v7[a4];
  if ( v7 != a3 )
  {
    while ( a1 != (_BYTE *)*v7 )
    {
      if ( a3 == ++v7 )
        goto LABEL_28;
    }
    return;
  }
LABEL_28:
  if ( (unsigned int)a4 < *(_DWORD *)(a2 + 16) )
  {
    v8 = (unsigned int)(a4 + 1);
    *(_DWORD *)(a2 + 20) = v8;
    *a3 = a1;
    ++*(_QWORD *)a2;
  }
  else
  {
LABEL_9:
    sub_C8CC70(a2, (__int64)a1, (__int64)a3, a4, a5, a6);
    if ( !v9 )
      return;
  }
  v10 = *(a1 - 16);
  if ( (v10 & 2) != 0 )
  {
    v12 = (_BYTE **)*((_QWORD *)a1 - 4);
    v11 = *((unsigned int *)a1 - 6);
  }
  else
  {
    v11 = (*((_WORD *)a1 - 8) >> 6) & 0xF;
    v12 = (_BYTE **)&a1[-8 * ((v10 >> 2) & 0xF) - 16];
  }
  v13 = &v12[v11];
  if ( v13 != v12 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v14 = *v12;
        if ( (unsigned __int8)(**v12 - 5) <= 0x1Fu )
          break;
LABEL_19:
        if ( v13 == ++v12 )
          return;
      }
      if ( *(_BYTE *)(a2 + 28) )
      {
        v15 = *(_QWORD **)(a2 + 8);
        v8 = *(unsigned int *)(a2 + 20);
        v11 = (__int64)&v15[v8];
        if ( v15 != (_QWORD *)v11 )
        {
          while ( v14 != (_BYTE *)*v15 )
          {
            if ( (_QWORD *)v11 == ++v15 )
              goto LABEL_24;
          }
          goto LABEL_19;
        }
LABEL_24:
        if ( (unsigned int)v8 >= *(_DWORD *)(a2 + 16) )
          goto LABEL_22;
        v8 = (unsigned int)(v8 + 1);
        ++v12;
        *(_DWORD *)(a2 + 20) = v8;
        *(_QWORD *)v11 = v14;
        ++*(_QWORD *)a2;
        if ( v13 == v12 )
          return;
      }
      else
      {
LABEL_22:
        ++v12;
        sub_C8CC70(a2, (__int64)v14, v11, v8, a5, a6);
        if ( v13 == v12 )
          return;
      }
    }
  }
}
