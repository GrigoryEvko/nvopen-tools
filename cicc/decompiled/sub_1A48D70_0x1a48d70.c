// Function: sub_1A48D70
// Address: 0x1a48d70
//
__int64 __fastcall sub_1A48D70(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 v5; // r8
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 *v8; // rax
  __int64 v9; // rcx
  unsigned __int64 v10; // rdx
  __int64 v11; // rdx

  v3 = *(_QWORD *)(a1 + 80);
  v4 = v3 + 8LL * *(unsigned int *)(a1 + 88);
  while ( v3 != v4 )
  {
    while ( 1 )
    {
      v5 = *(_QWORD *)(v4 - 8);
      if ( *(_BYTE *)(a2 + 16) > 0x10u )
        break;
      v4 -= 8;
      a2 = sub_15A46C0((unsigned int)*(unsigned __int8 *)(v5 + 16) - 24, (__int64 ***)a2, *(__int64 ***)v5, 0);
      if ( v3 == v4 )
        return a2;
    }
    v6 = sub_15F4880(*(_QWORD *)(v4 - 8));
    v7 = v6;
    if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
      v8 = *(__int64 **)(v6 - 8);
    else
      v8 = (__int64 *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
    if ( *v8 )
    {
      v9 = v8[1];
      v10 = v8[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v10 = v9;
      if ( v9 )
        *(_QWORD *)(v9 + 16) = *(_QWORD *)(v9 + 16) & 3LL | v10;
    }
    *v8 = a2;
    v11 = *(_QWORD *)(a2 + 8);
    v8[1] = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = (unsigned __int64)(v8 + 1) | *(_QWORD *)(v11 + 16) & 3LL;
    v4 -= 8;
    v8[2] = (a2 + 8) | v8[2] & 3;
    *(_QWORD *)(a2 + 8) = v8;
    a2 = v7;
    sub_15F2120(v7, *(_QWORD *)(a1 + 224));
  }
  return a2;
}
