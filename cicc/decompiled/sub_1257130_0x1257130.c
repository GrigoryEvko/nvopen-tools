// Function: sub_1257130
// Address: 0x1257130
//
__int64 __fastcall sub_1257130(char *a1)
{
  char v12; // [rsp+0h] [rbp-Ch] BYREF

  if ( a1 )
    *(_DWORD *)a1 = 0;
  else
    a1 = &v12;
  _RAX = 0;
  __asm { cpuid }
  if ( !(_DWORD)_RAX )
    return 0;
  _RAX = 0;
  __asm { cpuid }
  *(_DWORD *)a1 = _RAX;
  if ( !(_DWORD)_RAX )
    return 0;
  if ( (_DWORD)_RBX == 1970169159 )
    return ((_DWORD)_RDX == 1231384169) & (unsigned __int8)((_DWORD)_RCX == 1818588270);
  if ( (_DWORD)_RDX != 1769238117 || (_DWORD)_RBX != 1752462657 || (_DWORD)_RCX != 1145913699 )
    return 0;
  return 2;
}
