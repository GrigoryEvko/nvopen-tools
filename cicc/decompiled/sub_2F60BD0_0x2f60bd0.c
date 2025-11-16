// Function: sub_2F60BD0
// Address: 0x2f60bd0
//
__int64 __fastcall sub_2F60BD0(__int64 a1, unsigned int a2, __int64 a3)
{
  unsigned __int64 v3; // rbx
  __int64 result; // rax
  unsigned int *v5; // rax

  v3 = *(_QWORD *)(a1 + 128) + ((unsigned __int64)a2 << 6);
  result = *(unsigned __int8 *)(v3 + 57);
  if ( !*(_WORD *)(v3 + 57) && (unsigned int)(*(_DWORD *)v3 - 1) <= 1 )
  {
    v5 = *(unsigned int **)(v3 + 48);
    *(_BYTE *)(v3 + 58) = 1;
    result = sub_2F60BD0(a3, *v5, a1);
    *(_BYTE *)(v3 + 57) = result;
  }
  return result;
}
