// Function: sub_30F99C0
// Address: 0x30f99c0
//
__int64 __fastcall sub_30F99C0(__int64 a1, unsigned __int8 *a2)
{
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // rcx
  int v7; // eax
  int v8; // esi
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  unsigned int v13; // r15d
  __int64 v14; // rax
  int v15; // edi
  __int64 v16; // rdx
  __int64 v17; // r13
  __int64 v19; // rdi
  int v20; // eax
  int v21; // r8d
  __int64 v22[8]; // [rsp+0h] [rbp-80h] BYREF
  __int16 v23; // [rsp+40h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 40);
  v5 = *((_QWORD *)a2 - 4);
  v6 = *(_QWORD *)(v4 + 8);
  v7 = *(_DWORD *)(v4 + 24);
  if ( v7 )
  {
    v8 = v7 - 1;
    v9 = (v7 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( v5 == *v10 )
    {
LABEL_3:
      v12 = v10[1];
      if ( v12 )
        v5 = v12;
    }
    else
    {
      v20 = 1;
      while ( v11 != -4096 )
      {
        v21 = v20 + 1;
        v9 = v8 & (v20 + v9);
        v10 = (__int64 *)(v6 + 16LL * v9);
        v11 = *v10;
        if ( v5 == *v10 )
          goto LABEL_3;
        v20 = v21;
      }
    }
  }
  v13 = sub_B50F30((unsigned int)*a2 - 29, *(_QWORD *)(v5 + 8), *((_QWORD *)a2 + 1));
  if ( !(_BYTE)v13 )
    return sub_30F9620(a1, (__int64)a2);
  v14 = sub_B43CC0((__int64)a2);
  v15 = *a2;
  v16 = *((_QWORD *)a2 + 1);
  v22[0] = v14;
  memset(&v22[1], 0, 56);
  v23 = 257;
  v17 = sub_1002A60(v15 - 29, (unsigned __int8 *)v5, v16, v22);
  if ( !v17 )
    return sub_30F9620(a1, (__int64)a2);
  v19 = *(_QWORD *)(a1 + 40);
  v22[0] = (__int64)a2;
  *sub_FAA780(v19, v22) = v17;
  return v13;
}
