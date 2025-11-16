// Function: sub_22AF6E0
// Address: 0x22af6e0
//
void *__fastcall sub_22AF6E0(__int64 a1, __int64 *a2)
{
  __int64 v3; // rdx
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rdi
  void *result; // rax
  __int64 v11; // rdi
  void *v12; // rax
  __int64 v13; // rdx
  const void *v14; // rsi
  void *v15; // rax
  __int64 v16; // rdx
  const void *v17; // rsi
  void *v18; // rax
  __int64 v19; // rdx
  const void *v20; // rsi
  void *v21; // rax
  __int64 v22; // rdx
  const void *v23; // rsi

  v3 = a2[1];
  v5 = a2[2];
  v6 = *a2;
  *(_DWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = v3;
  *(_QWORD *)a1 = v6;
  *(_QWORD *)(a1 + 16) = v5;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  sub_C7D6A0(0, 0, 8);
  v7 = *((unsigned int *)a2 + 12);
  *(_DWORD *)(a1 + 48) = v7;
  if ( (_DWORD)v7 )
  {
    v12 = (void *)sub_C7D670(16 * v7, 8);
    v13 = *(unsigned int *)(a1 + 48);
    v14 = (const void *)a2[4];
    *(_QWORD *)(a1 + 32) = v12;
    *(_QWORD *)(a1 + 40) = a2[5];
    memcpy(v12, v14, 16 * v13);
  }
  else
  {
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 40) = 0;
  }
  *(_DWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  sub_C7D6A0(0, 0, 8);
  v8 = *((unsigned int *)a2 + 20);
  *(_DWORD *)(a1 + 80) = v8;
  if ( (_DWORD)v8 )
  {
    v21 = (void *)sub_C7D670(16 * v8, 8);
    v22 = *(unsigned int *)(a1 + 80);
    v23 = (const void *)a2[8];
    *(_QWORD *)(a1 + 64) = v21;
    *(_QWORD *)(a1 + 72) = a2[9];
    memcpy(v21, v23, 16 * v22);
  }
  else
  {
    *(_QWORD *)(a1 + 64) = 0;
    *(_QWORD *)(a1 + 72) = 0;
  }
  *(_DWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  sub_C7D6A0(0, 0, 4);
  v9 = *((unsigned int *)a2 + 28);
  *(_DWORD *)(a1 + 112) = v9;
  if ( (_DWORD)v9 )
  {
    v18 = (void *)sub_C7D670(8 * v9, 4);
    v19 = *(unsigned int *)(a1 + 112);
    v20 = (const void *)a2[12];
    *(_QWORD *)(a1 + 96) = v18;
    *(_QWORD *)(a1 + 104) = a2[13];
    memcpy(v18, v20, 8 * v19);
  }
  else
  {
    *(_QWORD *)(a1 + 96) = 0;
    *(_QWORD *)(a1 + 104) = 0;
  }
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 144) = 0;
  result = (void *)sub_C7D6A0(0, 0, 4);
  v11 = *((unsigned int *)a2 + 36);
  *(_DWORD *)(a1 + 144) = v11;
  if ( (_DWORD)v11 )
  {
    v15 = (void *)sub_C7D670(8 * v11, 4);
    v16 = *(unsigned int *)(a1 + 144);
    v17 = (const void *)a2[16];
    *(_QWORD *)(a1 + 128) = v15;
    *(_QWORD *)(a1 + 136) = a2[17];
    return memcpy(v15, v17, 8 * v16);
  }
  else
  {
    *(_QWORD *)(a1 + 128) = 0;
    *(_QWORD *)(a1 + 136) = 0;
  }
  return result;
}
