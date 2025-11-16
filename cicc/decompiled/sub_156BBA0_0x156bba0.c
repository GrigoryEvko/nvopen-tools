// Function: sub_156BBA0
// Address: 0x156bba0
//
__int64 __fastcall sub_156BBA0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _BYTE *a6, char a7)
{
  __int64 v8; // r13
  char v10; // r8
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rbx
  unsigned int v14; // r14d
  char v15; // r8
  unsigned int v16; // esi
  int v17; // edi
  unsigned int v18; // eax
  __int64 v19; // rcx
  unsigned int v20; // edx
  __int64 v21; // rax
  __int64 v23; // rax
  char *v26; // [rsp+20h] [rbp-150h] BYREF
  char v27; // [rsp+30h] [rbp-140h]
  char v28; // [rsp+31h] [rbp-13Fh]
  _DWORD v29[76]; // [rsp+40h] [rbp-130h] BYREF

  v8 = a2;
  v10 = a7;
  v11 = *(_QWORD *)(a4 + 24);
  if ( *(_DWORD *)(a4 + 32) > 0x40u )
    v11 = **(_QWORD **)(a4 + 24);
  v12 = *(_QWORD *)a2;
  v13 = *(_QWORD *)(*(_QWORD *)a2 + 32LL);
  v14 = v11 & (v13 - 1);
  if ( !a7 )
    v14 = v11;
  if ( v14 > 0x1F )
    return sub_15A06D0(v12);
  if ( v14 > 0x10 )
  {
    v14 -= 16;
    v23 = sub_15A06D0(v12);
    a3 = a2;
    v10 = a7;
    v8 = v23;
  }
  if ( (_DWORD)v13 )
  {
    v15 = v10 ^ 1;
    v16 = 0;
    v17 = -v14;
    do
    {
      v18 = v14;
      do
      {
        if ( v18 <= 0xF || (v20 = v13 - 16 + v18, !v15) )
          v20 = v18;
        v19 = v17 + v18++;
        v29[v19] = v16 + v20;
      }
      while ( v18 != v14 + 16 );
      v16 += 16;
      v17 += 16;
    }
    while ( v16 < (unsigned int)v13 );
  }
  v28 = 1;
  v26 = "palignr";
  v27 = 3;
  v21 = sub_156A7D0(a1, a3, v8, (__int64)v29, (unsigned int)v13, (__int64)&v26);
  return sub_156BB10(a1, a6, v21, a5);
}
