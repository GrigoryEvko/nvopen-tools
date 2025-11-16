// Function: sub_35994F0
// Address: 0x35994f0
//
__int64 __fastcall sub_35994F0(__int64 a1, __int64 a2, int a3, int a4)
{
  _QWORD *v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rsi
  __int64 v12; // r15
  __int64 v13; // rax
  unsigned int v14; // edx
  __int64 *v15; // rcx
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 (*v18)(); // rax
  int v20; // ecx
  unsigned int v21; // ecx
  unsigned int v22; // ebx
  unsigned __int64 v23; // rax
  int v24; // eax
  __int64 v25; // r10
  int v26; // r10d
  __int64 v27; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+10h] [rbp-50h]
  __int64 v29; // [rsp+18h] [rbp-48h]
  _BYTE v30[4]; // [rsp+28h] [rbp-38h] BYREF
  _DWORD v31[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v8 = sub_2E7B2C0(*(_QWORD **)(a1 + 8), a2);
  v11 = *(_QWORD *)(a1 + 136);
  v12 = (__int64)v8;
  v13 = *(unsigned int *)(a1 + 152);
  if ( !(_DWORD)v13 )
    goto LABEL_9;
  v10 = (unsigned int)(v13 - 1);
  v14 = v10 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v15 = (__int64 *)(v11 + 24LL * v14);
  v16 = *v15;
  if ( a2 != *v15 )
  {
    v20 = 1;
    while ( v16 != -4096 )
    {
      v26 = v20 + 1;
      v14 = v10 & (v20 + v14);
      v15 = (__int64 *)(v11 + 24LL * v14);
      v16 = *v15;
      if ( a2 == *v15 )
        goto LABEL_3;
      v20 = v26;
    }
    goto LABEL_9;
  }
LABEL_3:
  if ( v15 == (__int64 *)(v11 + 24 * v13) )
  {
LABEL_9:
    v21 = a3 - a4;
LABEL_10:
    sub_35990B0(a1, v12, a2, v21, v9, v10);
    return v12;
  }
  v17 = *(_QWORD *)(a1 + 32);
  v18 = *(__int64 (**)())(*(_QWORD *)v17 + 824LL);
  if ( v18 != sub_2FDC6B0 )
  {
    v28 = v15[1];
    v27 = v15[2];
    if ( ((unsigned __int8 (__fastcall *)(__int64, __int64, _BYTE *, _DWORD *))v18)(v17, a2, v30, v31) )
    {
      v22 = a3 - a4;
      v29 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL * v31[0] + 24);
      v23 = sub_35993A0(a1, v28);
      v24 = sub_3598DB0(*(_QWORD *)a1, v23);
      v25 = v29;
      v21 = v22;
      if ( v24 > a4 )
        v25 = v27 * v22 + v29;
      *(_QWORD *)(*(_QWORD *)(v12 + 32) + 40LL * v31[0] + 24) = v25;
      goto LABEL_10;
    }
  }
  return 0;
}
