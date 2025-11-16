// Function: sub_27940D0
// Address: 0x27940d0
//
__int64 __fastcall sub_27940D0(__int64 a1, __int64 a2, int a3, unsigned int a4, __int64 a5, __int64 a6)
{
  _QWORD **v9; // rdx
  int v10; // ecx
  __int64 *v11; // rax
  __int64 v12; // rax
  __int64 v13; // r8
  int v14; // eax
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 v17; // r8
  __int64 v18; // r9
  int v19; // r13d
  __int64 v20; // rax
  unsigned int *v21; // rax
  unsigned int v22; // edx
  unsigned int v23; // ecx
  int v25; // [rsp+Ch] [rbp-54h]
  const void *v26; // [rsp+10h] [rbp-50h]
  __int64 v28; // [rsp+18h] [rbp-48h]
  __int64 v29; // [rsp+28h] [rbp-38h]

  *(_QWORD *)(a1 + 16) = a1 + 32;
  v26 = (const void *)(a1 + 32);
  *(_DWORD *)a1 = -3;
  *(_BYTE *)(a1 + 4) = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 24) = 0x400000000LL;
  *(_QWORD *)(a1 + 48) = 0;
  v9 = *(_QWORD ***)(a5 + 8);
  v10 = *((unsigned __int8 *)v9 + 8);
  if ( (unsigned int)(v10 - 17) > 1 )
  {
    v12 = sub_BCB2A0(*v9);
  }
  else
  {
    BYTE4(v29) = (_BYTE)v10 == 18;
    LODWORD(v29) = *((_DWORD *)v9 + 8);
    v11 = (__int64 *)sub_BCB2A0(*v9);
    v12 = sub_BCE1B0(v11, v29);
  }
  v13 = a5;
  *(_QWORD *)(a1 + 8) = v12;
  v28 = a1 + 16;
  v14 = sub_2792F80(a2, v13);
  v16 = *(unsigned int *)(a1 + 24);
  if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
  {
    v25 = v14;
    sub_C8D5F0(v28, v26, v16 + 1, 4u, v16 + 1, v15);
    v16 = *(unsigned int *)(a1 + 24);
    v14 = v25;
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4 * v16) = v14;
  ++*(_DWORD *)(a1 + 24);
  v19 = sub_2792F80(a2, a6);
  v20 = *(unsigned int *)(a1 + 24);
  if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
  {
    sub_C8D5F0(v28, v26, v20 + 1, 4u, v17, v18);
    v20 = *(unsigned int *)(a1 + 24);
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4 * v20) = v19;
  v21 = *(unsigned int **)(a1 + 16);
  ++*(_DWORD *)(a1 + 24);
  v22 = *v21;
  v23 = v21[1];
  if ( *v21 > v23 )
  {
    *v21 = v23;
    v21[1] = v22;
    a4 = sub_B52F50(a4);
  }
  *(_BYTE *)(a1 + 4) = 1;
  *(_DWORD *)a1 = (a3 << 8) | a4;
  return a1;
}
