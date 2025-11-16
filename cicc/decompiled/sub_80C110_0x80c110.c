// Function: sub_80C110
// Address: 0x80c110
//
void __fastcall sub_80C110(int a1, __int64 *a2, _QWORD *a3)
{
  _QWORD *v4; // rdi
  __int64 v5; // rax

  if ( a1 )
  {
    v4 = (_QWORD *)qword_4F18BE0;
    ++*a3;
    v5 = v4[2];
    if ( (unsigned __int64)(v5 + 1) > v4[1] )
    {
      sub_823810(v4);
      v4 = (_QWORD *)qword_4F18BE0;
      v5 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v4[4] + v5) = 69;
    ++v4[2];
  }
  if ( a2 )
    sub_80C040(a2, a3);
}
