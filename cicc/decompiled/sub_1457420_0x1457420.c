// Function: sub_1457420
// Address: 0x1457420
//
_QWORD *__fastcall sub_1457420(__int64 a1, __int64 a2, __int64 a3, char a4, _QWORD *a5, __int64 a6)
{
  _QWORD *result; // rax
  _QWORD *v8; // rbx
  __int64 v9; // r12
  __int64 v10; // rdx
  __int64 v11; // rdx
  _QWORD *v12; // r15
  __int64 v13; // rsi
  _QWORD *v14; // r14
  _QWORD *v15; // r8
  _QWORD *v16; // r9
  _QWORD *v17; // r10
  unsigned int v18; // r11d
  _QWORD *v19; // rax
  _QWORD *v20; // rdi
  _QWORD *v21; // [rsp+8h] [rbp-38h]

  *(_QWORD *)(a1 + 32) = a1 + 64;
  *(_QWORD *)(a1 + 40) = a1 + 64;
  result = &a5[a6];
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a3;
  *(_BYTE *)(a1 + 16) = a4;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 48) = 4;
  *(_DWORD *)(a1 + 56) = 0;
  v21 = result;
  if ( result != a5 )
  {
    v8 = a5;
    v9 = a1 + 24;
    do
    {
      v10 = *v8;
      result = *(_QWORD **)(*v8 + 16LL);
      if ( result == *(_QWORD **)(*v8 + 8LL) )
        v11 = *(unsigned int *)(v10 + 28);
      else
        v11 = *(unsigned int *)(v10 + 24);
      v12 = &result[v11];
      if ( result != v12 )
      {
        while ( 1 )
        {
          v13 = *result;
          v14 = result;
          if ( *result < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v12 == ++result )
            goto LABEL_8;
        }
        if ( v12 != result )
        {
          v15 = *(_QWORD **)(a1 + 40);
          v16 = *(_QWORD **)(a1 + 32);
          if ( v15 != v16 )
          {
LABEL_12:
            sub_16CCBA0(v9, v13);
            v15 = *(_QWORD **)(a1 + 40);
            v16 = *(_QWORD **)(a1 + 32);
            goto LABEL_13;
          }
          while ( 1 )
          {
            v17 = &v15[*(unsigned int *)(a1 + 52)];
            v18 = *(_DWORD *)(a1 + 52);
            if ( v15 == v17 )
            {
LABEL_27:
              if ( v18 >= *(_DWORD *)(a1 + 48) )
                goto LABEL_12;
              *(_DWORD *)(a1 + 52) = v18 + 1;
              *v17 = v13;
              v16 = *(_QWORD **)(a1 + 32);
              ++*(_QWORD *)(a1 + 24);
              v15 = *(_QWORD **)(a1 + 40);
            }
            else
            {
              v19 = v15;
              v20 = 0;
              while ( *v19 != v13 )
              {
                if ( *v19 == -2 )
                  v20 = v19;
                if ( v17 == ++v19 )
                {
                  if ( !v20 )
                    goto LABEL_27;
                  *v20 = v13;
                  v15 = *(_QWORD **)(a1 + 40);
                  --*(_DWORD *)(a1 + 56);
                  v16 = *(_QWORD **)(a1 + 32);
                  ++*(_QWORD *)(a1 + 24);
                  break;
                }
              }
            }
LABEL_13:
            result = v14 + 1;
            if ( v14 + 1 == v12 )
              goto LABEL_8;
            v13 = *result;
            ++v14;
            if ( *result >= 0xFFFFFFFFFFFFFFFELL )
              break;
LABEL_17:
            if ( v12 == v14 )
              goto LABEL_8;
            if ( v15 != v16 )
              goto LABEL_12;
          }
          while ( v12 != ++result )
          {
            v13 = *result;
            v14 = result;
            if ( *result < 0xFFFFFFFFFFFFFFFELL )
              goto LABEL_17;
          }
        }
      }
LABEL_8:
      ++v8;
    }
    while ( v21 != v8 );
  }
  return result;
}
