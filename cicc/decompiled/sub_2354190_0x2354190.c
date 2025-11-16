// Function: sub_2354190
// Address: 0x2354190
//
__int64 __fastcall sub_2354190(unsigned __int64 *a1, __int64 *a2)
{
  __int64 v2; // r13
  unsigned int v3; // ebx
  __int64 v4; // r14
  __int64 v5; // r10
  __int64 v6; // r9
  __int64 v7; // r8
  __int64 v8; // rcx
  __int64 v9; // r15
  __int64 v10; // rax
  unsigned __int64 v11; // rdi
  __int64 v12; // r15
  _QWORD *v13; // r14
  _QWORD *v14; // r12
  __int64 v15; // rax
  __int64 v17; // [rsp+8h] [rbp-68h]
  __int64 v18; // [rsp+10h] [rbp-60h]
  __int64 v19; // [rsp+18h] [rbp-58h]
  __int64 v20; // [rsp+20h] [rbp-50h]
  __int64 v21; // [rsp+28h] [rbp-48h]
  __int64 v22; // [rsp+28h] [rbp-48h]
  unsigned __int64 v23[7]; // [rsp+38h] [rbp-38h] BYREF

  ++a2[6];
  v2 = a2[7];
  v3 = *((_DWORD *)a2 + 18);
  v4 = a2[8];
  a2[7] = 0;
  v5 = *a2;
  v6 = a2[1];
  a2[8] = 0;
  v7 = a2[2];
  v8 = a2[3];
  *((_DWORD *)a2 + 18) = 0;
  v9 = a2[5];
  v17 = v5;
  v18 = v6;
  v19 = v7;
  v20 = v8;
  v21 = a2[4];
  v10 = sub_22077B0(0x58u);
  if ( v10 )
  {
    *(_QWORD *)(v10 + 48) = v9;
    *(_QWORD *)(v10 + 64) = v2;
    *(_QWORD *)(v10 + 8) = v17;
    *(_QWORD *)(v10 + 16) = v18;
    *(_QWORD *)v10 = &unk_4A10178;
    *(_QWORD *)(v10 + 24) = v19;
    *(_QWORD *)(v10 + 32) = v20;
    *(_QWORD *)(v10 + 40) = v21;
    *(_QWORD *)(v10 + 56) = 1;
    *(_QWORD *)(v10 + 72) = v4;
    *(_DWORD *)(v10 + 80) = v3;
    v23[0] = v10;
    sub_2353900(a1, v23);
    v11 = v23[0];
    if ( !v23[0] )
      return sub_C7D6A0(0, 0, 8);
    v22 = 0;
    v2 = 0;
    v3 = 0;
  }
  else
  {
    v23[0] = 0;
    sub_2353900(a1, v23);
    v11 = v23[0];
    v22 = 72LL * v3;
    if ( !v23[0] )
      goto LABEL_5;
  }
  (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v11 + 8LL))(v11);
LABEL_5:
  if ( v3 )
  {
    v12 = v2;
    do
    {
      if ( *(_QWORD *)v12 != -8192 && *(_QWORD *)v12 != -4096 )
      {
        v13 = *(_QWORD **)(v12 + 8);
        v14 = &v13[3 * *(unsigned int *)(v12 + 16)];
        if ( v13 != v14 )
        {
          do
          {
            v15 = *(v14 - 1);
            v14 -= 3;
            if ( v15 != 0 && v15 != -4096 && v15 != -8192 )
              sub_BD60C0(v14);
          }
          while ( v13 != v14 );
          v14 = *(_QWORD **)(v12 + 8);
        }
        if ( v14 != (_QWORD *)(v12 + 24) )
          _libc_free((unsigned __int64)v14);
      }
      v12 += 72;
    }
    while ( v2 + v22 != v12 );
  }
  return sub_C7D6A0(v2, v22, 8);
}
