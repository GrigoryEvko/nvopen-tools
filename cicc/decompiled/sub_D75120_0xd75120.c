// Function: sub_D75120
// Address: 0xd75120
//
void __fastcall sub_D75120(__int64 *a1, __int64 *a2, char a3)
{
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // r9
  unsigned int v8; // eax
  __int64 v9; // r8
  __int64 v10; // rdi

  v4 = a2[8];
  v5 = *a1;
  v6 = *(_QWORD *)(v5 + 8);
  if ( v4 )
  {
    v7 = (unsigned int)(*(_DWORD *)(v4 + 44) + 1);
    v8 = *(_DWORD *)(v4 + 44) + 1;
  }
  else
  {
    v7 = 0;
    v8 = 0;
  }
  if ( v8 < *(_DWORD *)(v6 + 32) && *(_QWORD *)(*(_QWORD *)(v6 + 24) + 8 * v7) )
  {
    sub_D741C0((__int64)a1, a2, a3);
  }
  else
  {
    v9 = *(_QWORD *)(v5 + 128);
    v10 = (__int64)(a2 - 8);
    if ( *(_BYTE *)a2 == 26 )
      v10 = (__int64)(a2 - 4);
    sub_AC2B30(v10, v9);
  }
}
