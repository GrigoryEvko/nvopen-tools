// Function: sub_B809B0
// Address: 0xb809b0
//
__int64 *__fastcall sub_B809B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *result; // rax
  __int64 v4; // r8
  unsigned int v6; // ecx
  __int64 *v7; // rsi
  __int64 v8; // rdi
  _QWORD *v9; // r12
  __int64 v10; // rax
  _QWORD *v11; // rbx
  __int64 v12; // r14
  _QWORD *v13; // rcx
  __int64 v14; // rdx
  _QWORD *v15; // rax
  _QWORD *v16; // rdx
  __int64 v17; // rcx
  _QWORD *v18; // rax
  int v19; // esi
  int v20; // r10d

  result = (__int64 *)*(unsigned int *)(a1 + 248);
  v4 = *(_QWORD *)(a1 + 232);
  if ( (_DWORD)result )
  {
    v6 = ((_DWORD)result - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v7 = (__int64 *)(v4 + 104LL * v6);
    v8 = *v7;
    if ( a3 == *v7 )
    {
LABEL_3:
      result = (__int64 *)(v4 + 104LL * (_QWORD)result);
      if ( v7 != result )
      {
        v9 = (_QWORD *)v7[2];
        if ( *((_BYTE *)v7 + 36) )
          v10 = *((unsigned int *)v7 + 7);
        else
          v10 = *((unsigned int *)v7 + 6);
        v11 = &v9[v10];
        if ( v9 == v11 )
          goto LABEL_9;
        while ( *v9 >= 0xFFFFFFFFFFFFFFFELL )
        {
          if ( ++v9 == v11 )
            goto LABEL_9;
        }
        if ( v9 == v11 )
        {
LABEL_9:
          v12 = 0;
        }
        else
        {
          v16 = v9;
          v17 = 0;
          do
          {
            v18 = v16 + 1;
            if ( v16 + 1 == v11 )
            {
LABEL_28:
              v12 = v17 + 1;
              goto LABEL_10;
            }
            while ( 1 )
            {
              v16 = v18;
              if ( *v18 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v11 == ++v18 )
                goto LABEL_28;
            }
            ++v17;
          }
          while ( v18 != v11 );
          v12 = v17;
        }
LABEL_10:
        result = (__int64 *)*(unsigned int *)(a2 + 8);
        if ( (unsigned __int64)result + v12 > *(unsigned int *)(a2 + 12) )
        {
          sub_C8D5F0(a2, a2 + 16, (char *)result + v12, 8);
          result = (__int64 *)*(unsigned int *)(a2 + 8);
        }
        v13 = (_QWORD *)(*(_QWORD *)a2 + 8LL * (_QWORD)result);
        if ( v9 != v11 )
        {
          v14 = *v9;
          do
          {
            v15 = v9 + 1;
            *v13++ = v14;
            if ( v9 + 1 == v11 )
              break;
            while ( 1 )
            {
              v14 = *v15;
              v9 = v15;
              if ( *v15 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v11 == ++v15 )
                goto LABEL_17;
            }
          }
          while ( v15 != v11 );
LABEL_17:
          result = (__int64 *)*(unsigned int *)(a2 + 8);
        }
        *(_DWORD *)(a2 + 8) = (_DWORD)result + v12;
      }
    }
    else
    {
      v19 = 1;
      while ( v8 != -4096 )
      {
        v20 = v19 + 1;
        v6 = ((_DWORD)result - 1) & (v19 + v6);
        v7 = (__int64 *)(v4 + 104LL * v6);
        v8 = *v7;
        if ( a3 == *v7 )
          goto LABEL_3;
        v19 = v20;
      }
    }
  }
  return result;
}
