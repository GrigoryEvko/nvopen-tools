// Function: sub_CA0DC0
// Address: 0xca0dc0
//
__int64 __fastcall sub_CA0DC0(__int64 a1, _BYTE *a2, size_t a3, unsigned int *a4, unsigned int a5)
{
  __int64 result; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8

  sub_CA0C50(a1, a2, a3);
  *(_BYTE *)(a1 + 136) = 0;
  if ( a3 == 1 && *a2 == 45 )
  {
    *(_QWORD *)(a1 + 144) = sub_CB7210(a1);
    result = sub_2241E40(a1, a2, v9, v10, v11);
    *a4 = 0;
    *((_QWORD *)a4 + 1) = result;
  }
  else
  {
    sub_CB7060(a1 + 40, a2, a3, a4, a5);
    *(_BYTE *)(a1 + 136) = 1;
    *(_QWORD *)(a1 + 144) = a1 + 40;
    result = *a4;
    if ( (_DWORD)result )
      *(_BYTE *)(a1 + 32) = 1;
  }
  return result;
}
