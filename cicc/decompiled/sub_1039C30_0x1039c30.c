// Function: sub_1039C30
// Address: 0x1039c30
//
__int64 __fastcall sub_1039C30(__int64 *a1)
{
  __int64 v1; // rdx
  unsigned __int8 v2; // al
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 result; // rax

  v1 = *a1;
  v2 = *(_BYTE *)(*a1 - 16);
  if ( (v2 & 2) != 0 )
  {
    v3 = *(_QWORD *)(v1 - 32);
    v4 = *(unsigned int *)(v1 - 24);
  }
  else
  {
    v4 = (*(_WORD *)(v1 - 16) >> 6) & 0xF;
    v3 = v1 - 8LL * ((v2 >> 2) & 0xF) - 16;
  }
  v5 = *(_QWORD *)(v3 + 8 * v4 - 8);
  if ( *(_BYTE *)v5 != 1 || (v6 = *(_QWORD *)(v5 + 136), *(_BYTE *)v6 != 17) )
    BUG();
  result = *(_QWORD *)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    return *(_QWORD *)result;
  return result;
}
