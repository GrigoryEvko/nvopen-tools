// Function: sub_19E5420
// Address: 0x19e5420
//
__int64 __fastcall sub_19E5420(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rax
  char v6; // dl
  char v7; // si
  __int64 v8; // rdx
  __int64 v9; // rdi
  _QWORD *i; // rdx
  __int64 v11; // rcx

  v5 = sub_1412190(a2, a3);
  v7 = v6;
  v8 = *(_QWORD *)(a2 + 16);
  if ( v8 == *(_QWORD *)(a2 + 8) )
    v9 = *(unsigned int *)(a2 + 28);
  else
    v9 = *(unsigned int *)(a2 + 24);
  for ( i = (_QWORD *)(v8 + 8 * v9); i != v5; ++v5 )
  {
    if ( *v5 < 0xFFFFFFFFFFFFFFFELL )
      break;
  }
  v11 = *(_QWORD *)a2;
  *(_QWORD *)a1 = v5;
  *(_QWORD *)(a1 + 16) = a2;
  *(_QWORD *)(a1 + 8) = i;
  *(_QWORD *)(a1 + 24) = v11;
  *(_BYTE *)(a1 + 32) = v7;
  return a1;
}
