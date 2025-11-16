// Function: sub_2F3F610
// Address: 0x2f3f610
//
__int64 __fastcall sub_2F3F610(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // eax
  __int64 v7; // r15
  unsigned __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 result; // rax
  unsigned __int64 v13; // rsi
  __int64 v14; // r14
  __int64 v15; // rcx
  _QWORD *v16; // rax
  __int64 v17; // r12
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // r14
  __int64 v21; // rdi
  __int64 v22; // [rsp+8h] [rbp-38h]
  __int64 v23; // [rsp+8h] [rbp-38h]

  v6 = (a2 >> 31) ^ (2 * a2);
  v7 = v6;
  v10 = *(unsigned int *)(a1 + 80);
  if ( v6 >= (unsigned int)v10 )
  {
    v13 = v6 + 1;
    if ( v13 != v10 )
    {
      v14 = 8 * v13;
      if ( v13 < v10 )
      {
        v15 = *(_QWORD *)(a1 + 72);
        v19 = v15 + 8 * v10;
        v20 = v15 + v14;
        if ( v19 != v20 )
        {
          do
          {
            v21 = *(_QWORD *)(v19 - 8);
            v19 -= 8;
            if ( v21 )
            {
              v23 = v19;
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v21 + 16LL))(v21);
              v19 = v23;
            }
          }
          while ( v20 != v19 );
          goto LABEL_12;
        }
      }
      else
      {
        if ( v13 > *(unsigned int *)(a1 + 84) )
        {
          sub_2F3F540(a1 + 72, v13, v10, a4, a5, a6);
          v10 = *(unsigned int *)(a1 + 80);
        }
        v15 = *(_QWORD *)(a1 + 72);
        v16 = (_QWORD *)(v15 + 8 * v10);
        if ( v16 != (_QWORD *)(v15 + v14) )
        {
          do
          {
            if ( v16 )
              *v16 = 0;
            ++v16;
          }
          while ( (_QWORD *)(v15 + v14) != v16 );
LABEL_12:
          v15 = *(_QWORD *)(a1 + 72);
        }
      }
      *(_DWORD *)(a1 + 80) = v13;
      v11 = v15 + 8 * v7;
      result = *(_QWORD *)v11;
      if ( *(_QWORD *)v11 )
        return result;
      goto LABEL_14;
    }
  }
  v11 = *(_QWORD *)(a1 + 72) + 8LL * v6;
  result = *(_QWORD *)v11;
  if ( *(_QWORD *)v11 )
    return result;
LABEL_14:
  v17 = *(_QWORD *)a1;
  result = sub_22077B0(0x18u);
  if ( result )
  {
    v22 = result;
    sub_2F3F390(result, 4, v17);
    result = v22;
    *(_DWORD *)(v22 + 16) = a2;
    *(_QWORD *)v22 = &unk_4A2ABD0;
  }
  v18 = *(_QWORD *)v11;
  *(_QWORD *)v11 = result;
  if ( v18 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v18 + 16LL))(v18);
    return *(_QWORD *)v11;
  }
  return result;
}
