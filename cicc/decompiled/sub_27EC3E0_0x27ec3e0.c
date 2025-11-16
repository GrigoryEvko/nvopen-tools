// Function: sub_27EC3E0
// Address: 0x27ec3e0
//
__int64 __fastcall sub_27EC3E0(__int64 a1, __int64 a2)
{
  __int64 v3; // r9
  _QWORD *v4; // rdi
  __int64 v5; // rcx
  __int64 result; // rax
  int v7; // esi
  unsigned int v8; // edx
  __int64 v9; // r8
  __int64 v10; // rsi

  sub_31032E0(*(_QWORD *)(a1 + 120));
  v4 = *(_QWORD **)(a1 + 56);
  v5 = *(_QWORD *)(*v4 + 40LL);
  result = *(unsigned int *)(*v4 + 56LL);
  if ( (_DWORD)result )
  {
    v7 = result - 1;
    v8 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    result = v5 + 16LL * v8;
    v9 = *(_QWORD *)result;
    if ( a2 == *(_QWORD *)result )
    {
LABEL_3:
      v10 = *(_QWORD *)(result + 8);
      if ( v10 )
        return sub_D6E4B0(v4, v10, 0, v5, v9, v3);
    }
    else
    {
      result = 1;
      while ( v9 != -4096 )
      {
        v3 = (unsigned int)(result + 1);
        v8 = v7 & (result + v8);
        result = v5 + 16LL * v8;
        v9 = *(_QWORD *)result;
        if ( a2 == *(_QWORD *)result )
          goto LABEL_3;
        result = (unsigned int)v3;
      }
    }
  }
  return result;
}
