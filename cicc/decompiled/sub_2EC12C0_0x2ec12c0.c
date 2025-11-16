// Function: sub_2EC12C0
// Address: 0x2ec12c0
//
__int64 __fastcall sub_2EC12C0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rax
  __int64 v7; // rdi
  void (*v8)(); // rax
  __int64 result; // rax

  v6 = sub_2E88D60(a2);
  *(_WORD *)(a1 + 34) = 1;
  v7 = *(_QWORD *)(v6 + 16);
  v8 = *(void (**)())(*(_QWORD *)v7 + 336LL);
  if ( v8 != nullsub_1610 )
    ((void (__fastcall *)(__int64, __int64, _QWORD))v8)(v7, a1 + 32, a4);
  result = (unsigned int)dword_50218E8;
  switch ( dword_50218E8 )
  {
    case 1:
      *(_WORD *)(a1 + 34) = 1;
      break;
    case 2:
      *(_WORD *)(a1 + 34) = 256;
      break;
    case 3:
      *(_WORD *)(a1 + 34) = 0;
      return 0;
  }
  return result;
}
