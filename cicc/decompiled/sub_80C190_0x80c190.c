// Function: sub_80C190
// Address: 0x80c190
//
void __fastcall sub_80C190(char a1, _QWORD *a2)
{
  _QWORD *v3; // rdi
  __int64 v4; // rax
  _QWORD *v5; // rdi
  __int64 v6; // rax
  _QWORD *v7; // rdi
  __int64 v8; // rax

  if ( (a1 & 4) != 0 )
  {
    v3 = (_QWORD *)qword_4F18BE0;
    ++*a2;
    v4 = v3[2];
    if ( (unsigned __int64)(v4 + 1) > v3[1] )
    {
      sub_823810(v3);
      v3 = (_QWORD *)qword_4F18BE0;
      v4 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v3[4] + v4) = 114;
    ++v3[2];
  }
  if ( (a1 & 2) != 0 )
  {
    v5 = (_QWORD *)qword_4F18BE0;
    ++*a2;
    v6 = v5[2];
    if ( (unsigned __int64)(v6 + 1) > v5[1] )
    {
      sub_823810(v5);
      v5 = (_QWORD *)qword_4F18BE0;
      v6 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v5[4] + v6) = 86;
    ++v5[2];
  }
  if ( (a1 & 1) != 0 )
  {
    v7 = (_QWORD *)qword_4F18BE0;
    ++*a2;
    v8 = v7[2];
    if ( (unsigned __int64)(v8 + 1) > v7[1] )
    {
      sub_823810(v7);
      v7 = (_QWORD *)qword_4F18BE0;
      v8 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v7[4] + v8) = 75;
    ++v7[2];
  }
  if ( (a1 & 8) != 0 )
  {
    *a2 += 9LL;
    sub_8238B0(qword_4F18BE0, "U7_Atomic", 9);
  }
}
