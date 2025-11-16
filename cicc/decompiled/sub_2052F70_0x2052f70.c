// Function: sub_2052F70
// Address: 0x2052f70
//
bool __fastcall sub_2052F70(__int64 a1, __int64 a2)
{
  int *v2; // rax
  __int64 v3; // rcx
  __int64 v4; // rdx
  __int64 v5; // r8
  __int64 v6; // rdi
  __int64 v8; // rcx
  int *v9; // rax
  int v10; // edx

  v2 = *(int **)a2;
  if ( *(_QWORD *)(a2 + 8) - *(_QWORD *)a2 != 160 )
    return 1;
  v3 = *((_QWORD *)v2 + 1);
  v4 = *((_QWORD *)v2 + 11);
  v5 = *((_QWORD *)v2 + 13);
  v6 = *((_QWORD *)v2 + 3);
  if ( v3 == v4 )
    return v6 != v5 && (v3 != v6 || v3 != v5);
  if ( v4 == v6 && v3 == v5 )
    return 0;
  if ( v6 != v5 )
    return 1;
  v8 = (unsigned int)v2[20];
  if ( *v2 != (_DWORD)v8 || *(_BYTE *)(v6 + 16) > 0x10u )
    return 1;
  if ( !sub_1593BB0(v6, a2, v4, v8) )
    return 1;
  v9 = *(int **)a2;
  v10 = **(_DWORD **)a2;
  if ( v10 != 17 )
  {
    if ( v10 == 22 )
      return *((_QWORD *)v9 + 5) != *((_QWORD *)v9 + 16);
    return 1;
  }
  return *((_QWORD *)v9 + 4) != *((_QWORD *)v9 + 16);
}
