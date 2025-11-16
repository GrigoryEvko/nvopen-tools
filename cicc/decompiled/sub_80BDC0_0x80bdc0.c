// Function: sub_80BDC0
// Address: 0x80bdc0
//
__int64 __fastcall sub_80BDC0(unsigned __int64 a1, _QWORD *a2)
{
  _QWORD *v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rdx
  _QWORD *v6; // rdi
  __int64 result; // rax
  _BYTE v8[96]; // [rsp+0h] [rbp-60h] BYREF

  v3 = (_QWORD *)qword_4F18BE0;
  ++*a2;
  v4 = v3[2];
  if ( (unsigned __int64)(v4 + 1) > v3[1] )
  {
    sub_823810(v3);
    v3 = (_QWORD *)qword_4F18BE0;
    v4 = *(_QWORD *)(qword_4F18BE0 + 16);
  }
  *(_BYTE *)(v3[4] + v4) = 76;
  ++v3[2];
  ++*a2;
  sub_8238B0(v3, "i", 1);
  if ( a1 > 9 )
  {
    v5 = (int)sub_622470(a1, v8);
  }
  else
  {
    v8[1] = 0;
    v5 = 1;
    v8[0] = a1 + 48;
  }
  *a2 += v5;
  sub_8238B0(qword_4F18BE0, v8, v5);
  v6 = (_QWORD *)qword_4F18BE0;
  ++*a2;
  result = v6[2];
  if ( (unsigned __int64)(result + 1) > v6[1] )
  {
    sub_823810(v6);
    v6 = (_QWORD *)qword_4F18BE0;
    result = *(_QWORD *)(qword_4F18BE0 + 16);
  }
  *(_BYTE *)(v6[4] + result) = 69;
  ++v6[2];
  return result;
}
