// Function: sub_2A8AEC0
// Address: 0x2a8aec0
//
__int64 __fastcall sub_2A8AEC0(__int64 *a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // rcx
  __int64 v4; // rdx

  result = *((unsigned int *)a1 + 4);
  v3 = *a1;
  *((_DWORD *)a1 + 4) = 0;
  v4 = a1[1];
  *a1 = *a2;
  a1[1] = a2[1];
  *((_DWORD *)a1 + 4) = *((_DWORD *)a2 + 4);
  *a2 = v3;
  a2[1] = v4;
  *((_DWORD *)a2 + 4) = result;
  return result;
}
