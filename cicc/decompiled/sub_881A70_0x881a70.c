// Function: sub_881A70
// Address: 0x881a70
//
__int64 __fastcall sub_881A70(int a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v6; // r15
  char v7; // r14
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // r12
  unsigned int *i; // rax
  __int64 v14; // r14
  void *v15; // rax

  v6 = a3;
  v7 = a4;
  v8 = sub_823020(a1, 24, a3, a4, a5, a6);
  v11 = 1;
  *(_BYTE *)v8 = v6;
  v12 = v8;
  *(_BYTE *)(v8 + 1) = v7;
  *(_DWORD *)(v8 + 4) = a1;
  for ( i = (unsigned int *)&unk_3C1F2C4; ; ++i )
  {
    if ( a2 <= (unsigned int)v11 )
    {
      v14 = 8LL * (unsigned int)v11;
      goto LABEL_6;
    }
    if ( i == (unsigned int *)&unk_3C1F3A8 )
      break;
    v11 = *i;
  }
  v11 = 0;
  v14 = 0;
LABEL_6:
  *(_DWORD *)(v12 + 8) = v11;
  *(_DWORD *)(v12 + 12) = 0;
  v15 = (void *)sub_823020(a1, v14, v11, (__int64)&unk_3C1F3A8, v9, v10);
  *(_QWORD *)(v12 + 16) = v15;
  memset(v15, 0, v14);
  return v12;
}
