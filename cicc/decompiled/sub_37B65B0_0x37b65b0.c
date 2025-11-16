// Function: sub_37B65B0
// Address: 0x37b65b0
//
__int64 __fastcall sub_37B65B0(__int64 *a1)
{
  unsigned int v1; // edx
  unsigned int v2; // r9d
  int v3; // ecx
  __int64 v4; // r8
  __int64 v5; // rsi
  __int64 result; // rax
  int v7; // eax

  v1 = *(_DWORD *)a1;
  v2 = *((_DWORD *)a1 + 1);
  v3 = *((_DWORD *)a1 + 4);
  v4 = *a1;
  v5 = a1[1];
  while ( 1 )
  {
    result = *((unsigned int *)a1 - 5);
    if ( v1 >= (unsigned int)result && (v1 != (_DWORD)result || v2 >= *((_DWORD *)a1 - 4)) )
      break;
    *(_DWORD *)a1 = result;
    v7 = *((_DWORD *)a1 - 4);
    a1 = (__int64 *)((char *)a1 - 20);
    *((_DWORD *)a1 + 6) = v7;
    *((_DWORD *)a1 + 7) = *((_DWORD *)a1 + 2);
    *((_DWORD *)a1 + 8) = *((_DWORD *)a1 + 3);
    *((_DWORD *)a1 + 9) = *((_DWORD *)a1 + 4);
  }
  *a1 = v4;
  a1[1] = v5;
  *((_DWORD *)a1 + 4) = v3;
  return result;
}
