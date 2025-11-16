// Function: sub_823710
// Address: 0x823710
//
__int64 __fastcall sub_823710(int a1)
{
  __int64 v1; // rdx
  __int64 result; // rax
  __int64 v3; // rcx

  v1 = *((_QWORD *)qword_4F073B0 + a1);
  result = *(_QWORD *)(v1 + 16);
  if ( (unsigned __int64)(*(_QWORD *)(v1 + 24) - result) > 0x84F )
  {
    *(_QWORD *)(result + 24) = *(_QWORD *)(v1 + 24);
    v3 = qword_4F195E0;
    *(_QWORD *)(result + 32) = 0;
    *(_QWORD *)result = v3;
    *(_QWORD *)(result + 8) = result + 48;
    *(_QWORD *)(result + 16) = result + 48;
    *(_BYTE *)(result + 40) = 0;
    qword_4F195E0 = result;
    *(_QWORD *)(v1 + 24) = result;
  }
  *(_BYTE *)(v1 + 40) = 1;
  return result;
}
