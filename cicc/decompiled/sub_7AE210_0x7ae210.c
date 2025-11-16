// Function: sub_7AE210
// Address: 0x7ae210
//
__int64 __fastcall sub_7AE210(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx
  __int64 v3; // rcx
  int v4; // edx

  result = qword_4F08558;
  if ( qword_4F08558 )
    qword_4F08558 = *(_QWORD *)qword_4F08558;
  else
    result = sub_823970(112);
  *(_QWORD *)result = 0;
  *(_QWORD *)(result + 40) = 0;
  v2 = *(_QWORD *)&dword_4F063F8;
  *(_BYTE *)(result + 26) = 0;
  *(_QWORD *)(result + 28) = 0;
  *(_QWORD *)(result + 8) = v2;
  *(_QWORD *)(result + 16) = v2;
  *(_WORD *)(result + 24) = 9;
  v3 = *(_QWORD *)(a1 + 16);
  v4 = 0;
  if ( v3 )
    v4 = *(_DWORD *)(v3 + 28);
  *(_DWORD *)(result + 28) = v4;
  *(_DWORD *)(result + 32) = v4;
  if ( *(_QWORD *)(a1 + 8) )
    **(_QWORD **)(a1 + 16) = result;
  else
    *(_QWORD *)(a1 + 8) = result;
  *(_QWORD *)(a1 + 16) = result;
  return result;
}
