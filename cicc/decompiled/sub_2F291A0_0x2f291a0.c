// Function: sub_2F291A0
// Address: 0x2f291a0
//
__int64 __fastcall sub_2F291A0(__int64 a1, _DWORD *a2, _DWORD *a3)
{
  __int64 v3; // rcx
  __int64 v4; // rax
  _DWORD *v5; // rax

  v3 = *(_QWORD *)(a1 + 8);
  v4 = (unsigned int)(*(_DWORD *)(a1 + 16) + 2);
  *(_DWORD *)(a1 + 16) = v4;
  if ( (unsigned int)v4 >= (*(_DWORD *)(v3 + 40) & 0xFFFFFFu) )
    return 0;
  v5 = (_DWORD *)(*(_QWORD *)(v3 + 32) + 40 * v4);
  *a2 = v5[2];
  LODWORD(v5) = (*v5 >> 8) & 0xFFF;
  a2[1] = (_DWORD)v5;
  if ( (_DWORD)v5 )
    return 0;
  a3[1] = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL) + 40LL * (unsigned int)(*(_DWORD *)(a1 + 16) + 1) + 24);
  *a3 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL) + 8LL);
  return 1;
}
