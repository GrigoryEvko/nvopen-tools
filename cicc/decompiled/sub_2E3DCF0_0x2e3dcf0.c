// Function: sub_2E3DCF0
// Address: 0x2e3dcf0
//
__int64 __fastcall sub_2E3DCF0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v7; // eax
  __int64 v8; // rdi
  unsigned int v9; // r14d
  int v10; // ecx
  __int64 v11; // rsi
  int v12; // ecx
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // r8
  int v16; // eax
  unsigned __int64 v17; // rax
  int v19; // eax
  int v20; // r9d
  unsigned __int64 v21[7]; // [rsp+8h] [rbp-38h] BYREF

  v7 = sub_2E441D0(a4, a2, a3);
  v8 = *a1;
  v9 = v7;
  v10 = *(_DWORD *)(*a1 + 184);
  v11 = *(_QWORD *)(*a1 + 168);
  if ( !v10 )
  {
LABEL_7:
    v16 = -1;
    goto LABEL_4;
  }
  v12 = v10 - 1;
  v13 = v12 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v14 = (__int64 *)(v11 + 16LL * v13);
  v15 = *v14;
  if ( a2 != *v14 )
  {
    v19 = 1;
    while ( v15 != -4096 )
    {
      v20 = v19 + 1;
      v13 = v12 & (v19 + v13);
      v14 = (__int64 *)(v11 + 16LL * v13);
      v15 = *v14;
      if ( a2 == *v14 )
        goto LABEL_3;
      v19 = v20;
    }
    goto LABEL_7;
  }
LABEL_3:
  v16 = *((_DWORD *)v14 + 2);
LABEL_4:
  LODWORD(v21[0]) = v16;
  v21[0] = sub_FE8720(v8, (unsigned int *)v21);
  v17 = sub_1098D20(v21, v9);
  return sub_2E3D9B0(*a1, a3, v17);
}
