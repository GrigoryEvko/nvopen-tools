// Function: sub_2ABDA60
// Address: 0x2abda60
//
__int64 __fastcall sub_2ABDA60(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rax
  char v6; // dl
  char v7; // si
  __int64 v8; // rdi
  __int64 i; // rdx
  __int64 v10; // rcx

  v5 = sub_AE6EC0(a2, a3);
  v7 = v6;
  if ( *(_BYTE *)(a2 + 28) )
    v8 = *(unsigned int *)(a2 + 20);
  else
    v8 = *(unsigned int *)(a2 + 16);
  for ( i = *(_QWORD *)(a2 + 8) + 8 * v8; v5 != (_QWORD *)i; ++v5 )
  {
    if ( *v5 < 0xFFFFFFFFFFFFFFFELL )
      break;
  }
  v10 = *(_QWORD *)a2;
  *(_QWORD *)a1 = v5;
  *(_QWORD *)(a1 + 16) = a2;
  *(_QWORD *)(a1 + 8) = i;
  *(_QWORD *)(a1 + 24) = v10;
  *(_BYTE *)(a1 + 32) = v7;
  return a1;
}
