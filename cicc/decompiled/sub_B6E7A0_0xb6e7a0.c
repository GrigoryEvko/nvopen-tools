// Function: sub_B6E7A0
// Address: 0xb6e7a0
//
__int64 __fastcall sub_B6E7A0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r12
  _QWORD *v6; // rsi
  _QWORD *v7; // rdx
  _QWORD *v8; // rax
  __int64 v9; // rcx
  __int64 result; // rax
  __int64 v11; // rsi
  int v12; // ecx
  unsigned int v13; // edx
  __int64 v14; // rdi
  _QWORD *v15; // rax
  unsigned int v16; // r8d

  v5 = *a1;
  if ( *(_BYTE *)(*a1 + 28LL) )
  {
    v6 = *(_QWORD **)(v5 + 8);
    v7 = &v6[*(unsigned int *)(v5 + 20)];
    if ( v6 != v7 )
    {
      v8 = *(_QWORD **)(v5 + 8);
      while ( *v8 != a2 )
      {
        if ( v7 == ++v8 )
          goto LABEL_7;
      }
      v9 = (unsigned int)(*(_DWORD *)(v5 + 20) - 1);
      *(_DWORD *)(v5 + 20) = v9;
      *v8 = v6[v9];
      ++*(_QWORD *)v5;
      v5 = *a1;
    }
  }
  else
  {
    v15 = (_QWORD *)sub_C8CA60(*a1, a2, a3, a4);
    if ( v15 )
    {
      *v15 = -2;
      ++*(_DWORD *)(v5 + 24);
      ++*(_QWORD *)v5;
    }
    v5 = *a1;
  }
LABEL_7:
  result = *(unsigned int *)(v5 + 88);
  v11 = *(_QWORD *)(v5 + 72);
  if ( (_DWORD)result )
  {
    v12 = result - 1;
    v13 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    result = v11 + 16LL * v13;
    v14 = *(_QWORD *)result;
    if ( *(_QWORD *)result == a2 )
    {
LABEL_9:
      *(_QWORD *)result = -8192;
      --*(_DWORD *)(v5 + 80);
      ++*(_DWORD *)(v5 + 84);
    }
    else
    {
      result = 1;
      while ( v14 != -4096 )
      {
        v16 = result + 1;
        v13 = v12 & (result + v13);
        result = v11 + 16LL * v13;
        v14 = *(_QWORD *)result;
        if ( *(_QWORD *)result == a2 )
          goto LABEL_9;
        result = v16;
      }
    }
  }
  return result;
}
