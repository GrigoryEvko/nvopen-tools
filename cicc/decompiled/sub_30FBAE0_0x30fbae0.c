// Function: sub_30FBAE0
// Address: 0x30fbae0
//
__int64 *__fastcall sub_30FBAE0(__int64 *a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  unsigned int v9; // edx
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // r15

  v4 = *(_QWORD *)(a2 + 16);
  v5 = sub_B491C0((__int64)a3);
  v6 = sub_BC1CD0(v4, &unk_4F81450, v5);
  v7 = a3[5];
  if ( v7 )
  {
    v8 = (unsigned int)(*(_DWORD *)(v7 + 44) + 1);
    v9 = *(_DWORD *)(v7 + 44) + 1;
  }
  else
  {
    v8 = 0;
    v9 = 0;
  }
  if ( v9 < *(_DWORD *)(v6 + 40) && *(_QWORD *)(*(_QWORD *)(v6 + 32) + 8 * v8) )
  {
    *a1 = 0;
  }
  else
  {
    v11 = sub_30CC5F0(a2, (__int64)a3);
    v12 = sub_22077B0(0x40u);
    v13 = v12;
    if ( v12 )
      sub_30CABE0(v12, a2, a3, v11, 0);
    *a1 = v13;
  }
  return a1;
}
