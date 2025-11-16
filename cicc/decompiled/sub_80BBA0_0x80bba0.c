// Function: sub_80BBA0
// Address: 0x80bba0
//
__int64 __fastcall sub_80BBA0(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // rdi
  __int64 result; // rax

  if ( (unsigned int)sub_8D2FB0(a1) && dword_4D0425C && unk_4D04250 > 0x76BFu )
  {
    v2 = (_QWORD *)qword_4F18BE0;
    ++*a2;
    result = v2[2];
    if ( (unsigned __int64)(result + 1) > v2[1] )
    {
      sub_823810(v2);
      v2 = (_QWORD *)qword_4F18BE0;
      result = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v2[4] + result) = 90;
    ++v2[2];
  }
  else
  {
    *a2 += 2LL;
    return sub_8238B0(qword_4F18BE0, &unk_3C1BC40, 2);
  }
  return result;
}
