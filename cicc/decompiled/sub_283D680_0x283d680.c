// Function: sub_283D680
// Address: 0x283d680
//
_BYTE *__fastcall sub_283D680(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4)
{
  char *v6; // r13
  size_t v9; // rax
  _DWORD *v10; // rcx
  size_t v11; // rdx
  __int64 v12; // rdi
  __int64 (__fastcall *v13)(__int64); // rax
  _BYTE *result; // rax
  unsigned __int64 v15; // rdi
  char *v16; // rcx
  char *v17; // r13
  unsigned int v18; // ecx
  unsigned int v19; // ecx
  unsigned int v20; // eax
  __int64 v21; // rsi

  v6 = "loop-mssa(";
  if ( !a1[48] )
    v6 = "loop(";
  v9 = strlen(v6);
  v10 = *(_DWORD **)(a2 + 32);
  v11 = v9;
  if ( v9 <= *(_QWORD *)(a2 + 24) - (_QWORD)v10 )
  {
    if ( (unsigned int)v9 < 8 )
    {
      if ( (v9 & 4) != 0 )
      {
        *v10 = *(_DWORD *)v6;
        *(_DWORD *)((char *)v10 + (unsigned int)v9 - 4) = *(_DWORD *)&v6[(unsigned int)v9 - 4];
      }
      else if ( (_DWORD)v9 )
      {
        *(_BYTE *)v10 = *v6;
        if ( (v9 & 2) != 0 )
          *(_WORD *)((char *)v10 + (unsigned int)v9 - 2) = *(_WORD *)&v6[(unsigned int)v9 - 2];
      }
    }
    else
    {
      v15 = (unsigned __int64)(v10 + 2) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v10 = *(_QWORD *)v6;
      *(_QWORD *)((char *)v10 + (unsigned int)v9 - 8) = *(_QWORD *)&v6[(unsigned int)v9 - 8];
      v16 = (char *)v10 - v15;
      v17 = (char *)(v6 - v16);
      v18 = (v9 + (_DWORD)v16) & 0xFFFFFFF8;
      if ( v18 >= 8 )
      {
        v19 = v18 & 0xFFFFFFF8;
        v20 = 0;
        do
        {
          v21 = v20;
          v20 += 8;
          *(_QWORD *)(v15 + v21) = *(_QWORD *)&v17[v21];
        }
        while ( v20 < v19 );
      }
    }
    *(_QWORD *)(a2 + 32) += v11;
  }
  else
  {
    sub_CB6200(a2, (unsigned __int8 *)v6, v9);
  }
  v12 = *(_QWORD *)a1;
  v13 = *(__int64 (__fastcall **)(__int64))(**(_QWORD **)a1 + 24LL);
  if ( v13 == sub_2302CF0 )
    sub_283D540(v12 + 8, a2, a3, a4);
  else
    ((void (__fastcall *)(__int64, __int64, __int64, __int64))v13)(v12, a2, a3, a4);
  result = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 24) )
    return (_BYTE *)sub_CB5D20(a2, 41);
  *(_QWORD *)(a2 + 32) = result + 1;
  *result = 41;
  return result;
}
