// Function: sub_1319090
// Address: 0x1319090
//
__int64 __fastcall sub_1319090(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // rbx
  unsigned int *v5; // r12
  int v6; // ecx
  int v7; // edx
  unsigned int *v8; // rdi
  int v9; // edi
  int v10; // ecx
  unsigned int *v11; // rsi
  char *v12; // rdx
  int v13; // eax

  v4 = a1 + 80;
  sub_1317000(qword_4C6F230);
  sub_1317040(qword_4F96B78);
  v5 = dword_5260CA0;
  do
  {
    v6 = *(_DWORD *)(v4 + 4);
    v7 = *(_DWORD *)(v4 + 8);
    v8 = v5++;
    v4 += 28;
    sub_133DED0(v8, (unsigned int)((v7 << v6) + (1 << *(_DWORD *)(v4 - 28))));
  }
  while ( &dword_5260CA0[36] != v5 );
  v9 = dword_4F96B60;
  v10 = 78984;
  v11 = dword_5060A40;
  v12 = (char *)&unk_5260DF4;
  do
  {
    v13 = *(_DWORD *)v12;
    v12 += 40;
    *v11++ = v10;
    v9 += v13;
    v10 += 224 * v13;
  }
  while ( (char *)&unk_5260DE0 + 1460 != v12 );
  dword_4F96B60 = v9;
  return sub_130B2A0((__int64)&unk_5260B60, a2, a3, (__int64)&off_4C6F2E0);
}
