// Function: sub_1EBB180
// Address: 0x1ebb180
//
__int64 __fastcall sub_1EBB180(__int64 a1)
{
  __int64 v1; // rax
  int v2; // r15d
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v7; // [rsp+0h] [rbp-50h]
  __int64 v8; // [rsp+8h] [rbp-48h]
  _QWORD v9[7]; // [rsp+18h] [rbp-38h] BYREF

  v1 = *(_QWORD *)(a1 + 984);
  v9[0] = 0;
  v7 = *(_QWORD *)(v1 + 280);
  v8 = *(unsigned int *)(v1 + 288);
  if ( *(_DWORD *)(v1 + 288) )
  {
    v2 = 0;
    v3 = 0;
    do
    {
      v4 = v7 + 40 * v3;
      v5 = *(unsigned int *)(*(_QWORD *)v4 + 48LL);
      sub_16AF570(v9, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 848) + 376LL) + 8 * v5));
      if ( *(_BYTE *)(v4 + 32) && *(_BYTE *)(v4 + 33) && (*(_QWORD *)(v4 + 24) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        sub_16AF570(v9, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 848) + 376LL) + 8 * v5));
      v3 = (unsigned int)++v2;
    }
    while ( v2 != v8 );
  }
  return v9[0];
}
