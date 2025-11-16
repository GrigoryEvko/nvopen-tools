// Function: sub_2B68A00
// Address: 0x2b68a00
//
__int64 __fastcall sub_2B68A00(unsigned int **a1, unsigned int a2)
{
  unsigned int v2; // r12d
  unsigned int *v4; // rcx
  __int64 v5; // rax
  unsigned int *v6; // rdx
  __int64 v7; // rdi
  _BYTE *v8; // r13
  unsigned int v9; // eax
  __int64 *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  _BYTE *v14; // rax
  __int64 v15[2]; // [rsp+0h] [rbp-40h] BYREF
  _BYTE *v16[6]; // [rsp+10h] [rbp-30h] BYREF

  v2 = 1;
  v4 = a1[1];
  v5 = *(_QWORD *)(*(_QWORD *)v4 + 48LL * a2) + 16LL * **a1;
  v6 = a1[2];
  if ( *(_BYTE *)(v5 + 8) != *(_BYTE *)v6 )
    return v2;
  v2 = *(unsigned __int8 *)(v5 + 9);
  if ( (_BYTE)v2 )
    return v2;
  v7 = *((_QWORD *)v4 + 31);
  v8 = *(_BYTE **)v5;
  if ( v7 )
  {
    v9 = sub_D48480(v7, *(_QWORD *)v5, (__int64)v6, (__int64)v4);
    if ( (_BYTE)v9 )
      return v9;
    v4 = a1[1];
  }
  v11 = (__int64 *)*((_QWORD *)v4 + 27);
  v12 = *(_QWORD *)a1[3];
  v15[1] = (__int64)v8;
  v15[0] = v12;
  if ( !sub_2B5F980(v15, 2u, v11) || !v13 )
    return v2;
  v14 = *(_BYTE **)a1[3];
  v16[1] = v8;
  v16[0] = v14;
  return (unsigned int)sub_2B17600(v16, 2);
}
