// Function: sub_80BEC0
// Address: 0x80bec0
//
__int64 __fastcall sub_80BEC0(__int64 a1, int a2, _QWORD *a3)
{
  unsigned __int64 v3; // r12
  _QWORD *v5; // rdi
  __int64 v6; // rax
  __int64 result; // rax
  __int64 v8; // rax
  int v9; // eax
  __int64 v10; // rdi
  _QWORD *v11; // rdi
  int v12; // eax
  __int64 v13; // rdi
  _BYTE v14[96]; // [rsp+0h] [rbp-60h] BYREF

  v3 = a1 - 2;
  if ( a2 )
  {
    v5 = (_QWORD *)qword_4F18BE0;
    ++*a3;
    v6 = v5[2];
    if ( (unsigned __int64)(v6 + 1) > v5[1] )
    {
      sub_823810(v5);
      v5 = (_QWORD *)qword_4F18BE0;
      v6 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v5[4] + v6) = 95;
    ++v5[2];
    if ( v3 <= 9 )
      goto LABEL_5;
    ++*a3;
    v8 = v5[2];
    if ( (unsigned __int64)(v8 + 1) > v5[1] )
    {
      sub_823810(v5);
      v5 = (_QWORD *)qword_4F18BE0;
      v8 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v5[4] + v8) = 95;
    ++v5[2];
    v9 = sub_622470(v3, v14);
    v10 = qword_4F18BE0;
    *a3 += v9;
    sub_8238B0(v10, v14, v9);
    v11 = (_QWORD *)qword_4F18BE0;
    ++*a3;
    result = v11[2];
    if ( (unsigned __int64)(result + 1) > v11[1] )
    {
      sub_823810(v11);
      v11 = (_QWORD *)qword_4F18BE0;
      result = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v11[4] + result) = 95;
    ++v11[2];
  }
  else
  {
    if ( v3 <= 9 )
    {
      v5 = (_QWORD *)qword_4F18BE0;
LABEL_5:
      ++*a3;
      v14[0] = v3 + 48;
      v14[1] = 0;
      return sub_8238B0(v5, v14, 1);
    }
    v12 = sub_622470(a1 - 2, v14);
    v13 = qword_4F18BE0;
    *a3 += v12;
    return sub_8238B0(v13, v14, v12);
  }
  return result;
}
