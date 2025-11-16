// Function: sub_811640
// Address: 0x811640
//
__int64 __fastcall sub_811640(__int64 a1, _QWORD *a2)
{
  _QWORD *v4; // rdi
  __int64 v5; // rax
  unsigned int v6; // esi
  int v7; // edx
  _QWORD *v8; // rdi
  __int64 result; // rax
  int v10[5]; // [rsp+Ch] [rbp-14h] BYREF

  v4 = (_QWORD *)qword_4F18BE0;
  ++*a2;
  v10[0] = 0;
  v5 = v4[2];
  if ( (unsigned __int64)(v5 + 1) > v4[1] )
  {
    sub_823810(v4);
    v4 = (_QWORD *)qword_4F18BE0;
    v5 = *(_QWORD *)(qword_4F18BE0 + 16);
  }
  *(_BYTE *)(v4[4] + v5) = 90;
  ++v4[2];
  if ( sub_80A070(a1, v10) )
  {
    v6 = v10[0];
    v7 = 0;
  }
  else
  {
    v10[0] = 1;
    v6 = 1;
    v7 = 1;
  }
  sub_8111C0(a1, v6, v7, 1, 0, 0, (__int64)a2);
  v8 = (_QWORD *)qword_4F18BE0;
  ++*a2;
  result = v8[2];
  if ( (unsigned __int64)(result + 1) > v8[1] )
  {
    sub_823810(v8);
    v8 = (_QWORD *)qword_4F18BE0;
    result = *(_QWORD *)(qword_4F18BE0 + 16);
  }
  *(_BYTE *)(v8[4] + result) = 69;
  ++v8[2];
  return result;
}
