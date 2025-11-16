// Function: sub_20F7AB0
// Address: 0x20f7ab0
//
__int64 __fastcall sub_20F7AB0(unsigned int *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  unsigned int v6; // ecx
  _WORD *v7; // r10
  __int64 v8; // rdx
  unsigned __int16 i; // si
  __int64 v10; // rcx
  __int64 result; // rax

  ++a1[1];
  *((_QWORD *)a1 + 5) = 0;
  if ( !a3 )
    BUG();
  v5 = *a1;
  v6 = *(_DWORD *)(*(_QWORD *)(a3 + 8) + 24 * v5 + 16);
  v7 = (_WORD *)(*(_QWORD *)(a3 + 56) + 2LL * (v6 >> 4));
  v8 = 0;
  for ( i = *v7 + v5 * (v6 & 0xF); ; i += result )
  {
    v10 = (unsigned int)v8++;
    *(_DWORD *)(*((_QWORD *)a1 + 6) + 112 * v10 + 88) = *(_DWORD *)(a2 + 216LL * i);
    result = (unsigned __int16)v7[v8];
    if ( !(_WORD)result )
      break;
  }
  return result;
}
