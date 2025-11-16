// Function: sub_103DAF0
// Address: 0x103daf0
//
__int64 __fastcall sub_103DAF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 result; // rax
  int v6; // edi
  unsigned int v8; // edx
  __int64 v9; // r8
  __int64 v10; // r13
  _WORD *v11; // rdx
  unsigned int v12; // r9d

  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(_QWORD *)(v3 + 40);
  result = *(unsigned int *)(v3 + 56);
  if ( (_DWORD)result )
  {
    v6 = result - 1;
    v8 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    result = v4 + 16LL * v8;
    v9 = *(_QWORD *)result;
    if ( a2 == *(_QWORD *)result )
    {
LABEL_3:
      v10 = *(_QWORD *)(result + 8);
      if ( v10 )
      {
        v11 = *(_WORD **)(a3 + 32);
        if ( *(_QWORD *)(a3 + 24) - (_QWORD)v11 <= 1u )
        {
          a3 = sub_CB6200(a3, (unsigned __int8 *)"; ", 2u);
        }
        else
        {
          *v11 = 8251;
          *(_QWORD *)(a3 + 32) += 2LL;
        }
        sub_103D830(v10, a3);
        result = *(_QWORD *)(a3 + 32);
        if ( *(_QWORD *)(a3 + 24) == result )
        {
          return sub_CB6200(a3, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *(_BYTE *)result = 10;
          ++*(_QWORD *)(a3 + 32);
        }
      }
    }
    else
    {
      result = 1;
      while ( v9 != -4096 )
      {
        v12 = result + 1;
        v8 = v6 & (result + v8);
        result = v4 + 16LL * v8;
        v9 = *(_QWORD *)result;
        if ( a2 == *(_QWORD *)result )
          goto LABEL_3;
        result = v12;
      }
    }
  }
  return result;
}
