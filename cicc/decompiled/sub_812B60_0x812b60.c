// Function: sub_812B60
// Address: 0x812b60
//
_QWORD *__fastcall sub_812B60(unsigned int *a1, __int64 a2, _QWORD *a3)
{
  _QWORD *v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rax
  _QWORD *result; // rax
  __int64 v9; // rdx
  int v10; // eax
  _QWORD v11[12]; // [rsp+0h] [rbp-60h] BYREF

  v5 = (_QWORD *)qword_4F18BE0;
  ++*a3;
  v6 = v5[2];
  if ( (unsigned __int64)(v6 + 1) > v5[1] )
  {
    sub_823810(v5);
    v5 = (_QWORD *)qword_4F18BE0;
    v6 = *(_QWORD *)(qword_4F18BE0 + 16);
  }
  *(_BYTE *)(v5[4] + v6) = 84;
  ++v5[2];
  v7 = *a1;
  if ( (_DWORD)v7 != 1 )
  {
    if ( (unsigned __int64)(v7 - 2) > 9 )
    {
      v10 = sub_622470(v7 - 2, v11);
      v5 = (_QWORD *)qword_4F18BE0;
      v9 = v10;
    }
    else
    {
      v9 = 1;
      LOWORD(v11[0]) = (unsigned __int8)(v7 - 2 + 48);
    }
    *a3 += v9;
    sub_8238B0(v5, v11, v9);
    v5 = (_QWORD *)qword_4F18BE0;
  }
  ++*a3;
  result = (_QWORD *)v5[2];
  if ( (unsigned __int64)result + 1 > v5[1] )
  {
    sub_823810(v5);
    v5 = (_QWORD *)qword_4F18BE0;
    result = *(_QWORD **)(qword_4F18BE0 + 16);
  }
  *((_BYTE *)result + v5[4]) = 95;
  ++v5[2];
  if ( a2 )
  {
    v11[0] = a2;
    return sub_811CB0(v11, 0, 0, a3);
  }
  return result;
}
